# To consider — performance & correctness notes

Findings from a profiling session on 2026-07-07.  Nothing here is done yet;
this is a backlog of candidate optimisations and one latent bug.

---

## Profile (baseline)

Per-generation component timings, CPU, default `SimConfig`
(P=64, L=41, S=8, n_layers=3, n_steps=20):

| Component                   | ms/gen | share |
|-----------------------------|-------:|------:|
| `route_signals`             |   1178 | 45.9% |
| `forward`                   |    754 | 29.4% |
| `find_division_candidates`  |    552 | 21.5% |
| `score_plants`              |     61 |  2.4% |
| `apply_divisions`           |     12 |  0.5% |
| `resolve_race_conditions`   |      7 |  0.3% |
| `apply_budget_clip`         |    0.1 |  0.0% |

The ranking looks the same on GPU (route and forward are both the largest;
`find_div` is likely overhead-bound there too).  Everything below `find_div`
(<3% combined) is not worth optimising.

---

## 1. `forward` (29%) — already optimal, leave it

Benchmarked a batched-matmul (`bmm`) reformulation against the current
`torch.einsum('pijab,pcadb->pijcd', ...)`: bit-identical output, but
**0.98–1.00×** on CPU/GPU (slightly *worse* on CPU).  PyTorch already routes
this einsum to the same BLAS GEMM path.  The op is at its FLOP floor
(~`P·L²·S²·25` MACs/layer) — no algorithmic headroom.  **Do not touch.**

---

## 2. `route_signals` (46%) — still #1, headroom remains

The shift + `scatter_add_` rewrite (commit `9b201dd`) removed the masked
`index_put_`, but two inefficiencies remain:

- The `gather` and `scatter_add_` both expand their index over the `n_signal`
  axis, although the slot mapping depends only on orientation
  (`ABS_TO_REL[orient, ...]`), **not** on the signal.  We do `P·L²·S·4`
  indexed reads/writes when the routing pattern is identical across all `S`
  signals.
- It runs 4 separate (gather → roll → scatter) passes ≈ 12 kernel launches.

**Idea:** route into an *absolute-face* intermediate `arr[p,i,j,s,face]` using
4 pure dense shifts + adds (arrival face `a_neg` is a constant per direction, so
no scatter is needed), then do the relative↔absolute slot remap **once** as a
size-4 permutation.  This collapses the per-signal scatter into a single
small-axis remap.  Bigger potential win, but it's the hottest path — needs a
careful equivalence check (reuse the CPU+CUDA allclose harness from the last
route rewrite).

---

## 3. `find_division_candidates` (21.5%) — untouched, best effort-to-payoff

~28 ms/call to process ~430k elements → allocation/overhead-bound.  It:
- rebuilds `arange().view().expand()` coordinate grids on **every** call (the
  same grids are also rebuilt in `route_signals`),
- materialises ~8 separate `(P, L, L, 4)` intermediates,
- does an `alive[p_idx, dest_i_c, dest_j_c]` advanced-indexing gather to check
  destination emptiness.

**Wins:**
- Cache the coordinate grids module-level (keyed by shape/device) instead of
  rebuilding each step; share them with `route_signals`.
- Replace the destination-emptiness gather with 4 dense shifts of the `alive`
  mask (same `_shift2d` trick), avoiding the advanced-indexing gather.

Self-contained and low-risk; recommended **first** target.

---

## 4. `resolve_race_conditions` — latent correctness bug (not perf)

The tie-break key `flat_target * 1e12 - value.double()` overflows `double`'s
2⁵³ exact-integer range once `flat_target·1e12 > ~9e15`, i.e. for
`flat_target > ~9000` (easily reached: `flat_target = p·L² + i·L + j`).  Beyond
that, close `value`s can't be distinguished and the race winner becomes
arbitrary.

**Fix:** use a proper `scatter_reduce(..., reduce="amax")` over `flat_target`
(or lexicographic sort on the pair without the magic multiplier).  Worth doing
regardless of performance.

---

## Suggested order

1. `find_division_candidates` (clear redundant work, ~21%, low risk)
2. `route_signals` absolute-intermediate reformulation (hottest path, needs
   equivalence testing)
3. `resolve_race_conditions` bug fix (correctness)

Each should ship with a benchmark + allclose/behaviour check against the current
implementation, as done for the previous `route_signals` rewrite.

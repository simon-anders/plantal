# CLAUDE.md — Artificial Plant Life Simulation

This file gives a Claude Code session everything it needs to continue development
without access to the conversation history in which this code was designed.

---

## What this project is

A PyTorch simulation of evolving artificial plants, implemented as a cellular
automaton on a square grid.  Each "plant" grows from a single seed cell by
dividing into neighbouring grid squares.  Plants are evolved by a simple
mutation-selection loop that rewards maximising exposed boundary length
(fractal-like structures are expected to be optimal).

The full human-readable specification is in `artificial_plant_life.md`.
Read it before making any non-trivial change to the simulation logic.

---

## File map

```
constants.py   Named constants and look-up tables (permutations, grid deltas)
world.py       SimConfig + SimState dataclasses + init_sim
network.py     Network dataclass + init_network + forward pass
routing.py     apply_budget_clip + route_signals
division.py    find_division_candidates + resolve_race_conditions + apply_divisions
scoring.py     flood_fill_empty + flood_fill_live + score_plants
               + score_boundary_sides + make_size_penalised_score
evolution.py   step + run_generation + select_and_reproduce + run_evolution
               + default_callback + make_alive_viewer
tests.py       17 unit tests (run with: python tests.py)
artificial_plant_life.md   Full specification
```

---

## Key data structures

### `SimConfig`  (world.py)
A frozen dataclass holding all static simulation/evolution hyperparameters,
threaded through the orchestration layer (`run_evolution` / `run_generation` /
`select_and_reproduce`), which unpack the primitives their leaf functions need.
```
n_plants=64  l_world=41  n_signal=8  n_layers=3  n_steps=20
thr_division=0.5  signal_max=inf  sd_mut=0.05  device=cpu
```
`__post_init__` asserts `n_signal >= 3` and `l_world >= 2*n_steps + 1`.
Run-control settings (`n_generations`, `callback`) are deliberately kept out of
`SimConfig` and passed to `run_evolution` directly.

### `SimState`  (world.py)
```
alive       BoolTensor   (n_plants, l_world, l_world)
orientation LongTensor   (n_plants, l_world, l_world)   absolute dir of apex
output      FloatTensor  (n_plants, l_world, l_world, n_signal, 5)
```
`output` doubles as the current input tensor: after routing it holds the
signals each cell will receive at the next forward pass.

### `Network`  (network.py)
```
weights  FloatTensor  (n_plants, n_layers, n_signal, n_signal, 5, 5)
biases   FloatTensor  (n_plants, n_layers, n_signal, 5)
```
Each plant has its own weights; all cells within a plant share the same weights.

---

## Named constants  (constants.py)

### Signal indices  (n_signal >= 3 is a hard requirement)
```python
DIVISION_TRIGGER   = 0   # directional output triggers cell division
INITIAL_SIGNAL     = 1   # seeded to 1 in the seed cell at t=0; no other role
DIVISION_DIRECTION = 2   # argmax determines daughter cell's orientation
```

### Relative direction indices  (last axis of the (n_signal, 5) tensor)
```python
APEX=0, BASE=1, LEFT=2, RIGHT=3, INTERNAL=4
```

### Absolute direction indices
```python
NORTH=0, SOUTH=1, WEST=2, EAST=3
```

### Rotation look-up tables
```python
REL_TO_ABS[orientation, rel_dir] -> abs_dir   # (4,4) LongTensor
ABS_TO_REL[orientation, abs_dir] -> rel_dir   # (4,4) LongTensor
NEGATE_DIR[abs_dir]              -> opposite  # (4,)  LongTensor  [1,0,3,2]
DIR_TO_DELTA[abs_dir]            -> (di, dj)  # (4,2) LongTensor
```
`REL_TO_ABS` and `ABS_TO_REL` are mutually inverse for each orientation.
`NEGATE_DIR` is self-inverse.  These invariants are verified in `tests.py`.

---

## Step execution order  (evolution.py: `step`)

```
1. forward(network, state.output)          # n_layers dense layers, ReLU after each
2. apply_budget_clip(output, alive, ...)   # scale down cells over signal_max
3. find_division_candidates(...)           # check DIVISION_TRIGGER > thr_division
   resolve_race_conditions(...)            # keep highest signal per target square
   apply_divisions(state, candidates)      # update alive, orientation, output
4. route_signals(output, alive, orient)    # scatter directional outputs to neighbours
                                           # INTERNAL stays in place
                                           # dead cells' inputs zeroed at end
```

---

## Signal routing  (routing.py)

For each live cell and each relative direction r ∈ {APEX, BASE, LEFT, RIGHT}:
1. `abs_dir  = REL_TO_ABS[sender_orientation, r]`
2. `arr_abs  = NEGATE_DIR[abs_dir]`          (face the signal arrives at)
3. Shift signal to grid position `+ DIR_TO_DELTA[abs_dir]`
4. `rel_slot = ABS_TO_REL[receiver_orientation, arr_abs]`
5. Accumulate into `new_input[..., rel_slot]`

Signals to dead squares survive the scatter but are erased by the dead-cell
masking at the end of `route_signals`.  Because division (step 3) runs before
routing (step 4), a newly born daughter cell IS alive during routing and will
correctly receive signals from its neighbours in its birth step.

**Implementation** (rewritten for speed; ~1.6× CPU / ~3.3–3.7× CUDA vs. the
old version, bit-identical output): the scatter is reformulated by **absolute
direction of travel** a ∈ {N,S,W,E} instead of relative direction.  For a fixed
`a` every cell performs the same grid shift, so each of the 4 iterations is:
1. gather the signal each cell sends in direction `a` from slot
   `ABS_TO_REL[sender_orientation, a]` (the inverse of REL_TO_ABS),
2. zero dead senders, then shift the whole field one step via `torch.roll`
   (edge-zeroed by `_shift2d`, no wrap-around),
3. `scatter_add_` into slot `ABS_TO_REL[dest_orientation, NEGATE_DIR[a]]` of
   `new_input`'s size-5 relative-slot axis.

This replaced the previous boolean-masked `index_put_(accumulate=True)` scatter.
The per-cell mapping above is unchanged; only the loop axis and scatter differ.

---

## Division mechanics  (division.py)

- Candidates: `output[:,:, DIVISION_TRIGGER, 0:4] > thr_division` AND target empty
- Per cell: only the **strongest** directional component fires
- Race condition (two cells → same square): highest signal value wins
- All candidates evaluated against the alive tensor **at start of step**;
  all surviving divisions applied as a **batch update**
- Mother: `output[:,:,:, INTERNAL] /= 2`  (all n_signal signals)
- Daughter: `output[:,:,:, INTERNAL] = mother post-halved`; all other positions = 0
- Budget is NOT re-checked after division
- Daughter orientation: `argmax(output[:,:, DIVISION_DIRECTION, 0:4])` in
  absolute direction space (via REL_TO_ABS); tie / all-zero → inherit mother

---

## Scoring  (scoring.py)

Scoring is **pluggable**: `run_evolution(score_fn=...)` takes any
`score_fn(alive, l_world) -> Tensor` and evaluates it on each generation's
final state.  Two scorers are provided; both use the same dual flood fill and
both keep only **seed-connected** live cells:
- `flood_fill_live`: seed-connected component of live cells (flood fill from
  seed through live cells).
- `flood_fill_empty`: empty cells reachable from the world border (flood fill
  from border through empty cells).
- Both flood fills use iterated 2D convolution with a 3×3 cross stencil.

### `score_plants`  (default) — counts boundary **cells**
Score = number of seed-connected live cells adjacent to border-reachable empty
space (dilate `flood_fill_empty` by one step, intersect with `alive`).

### `score_boundary_sides` — counts boundary **sides** (edges)
For each seed-connected live cell, count how many of its 4 neighbours are
border-reachable empty cells, and sum.  A cell adjacent to 2 such empties adds
2; two live cells adjacent to the same empty each add 1.  Rewards total exposed
perimeter length.  (Reuses `_CROSS`; the centre term is 0 at live cells since a
live cell is never itself empty, so no separate 4-neighbour kernel is needed.)

### `cell_penalty` and `make_size_penalised_score`
Both scorers accept an optional `cell_penalty: float = 0.0`.  When > 0,
`cell_penalty * (number of alive cells)` is subtracted — **all** alive cells
count (via `alive.sum()`), so detached fragments are penalised but not
rewarded.  `cell_penalty == 0` returns a `LongTensor` (unchanged); a nonzero
penalty returns a `FloatTensor`.

`make_size_penalised_score(base_score_fn=score_plants, cell_penalty=0.0)` is a
factory returning a ready-to-use `score_fn(alive, l_world)` with the penalty
baked in, e.g.:
```python
run_evolution(cfg, n_generations,
    score_fn=make_size_penalised_score(score_boundary_sides, cell_penalty=0.5))
```

This rewards large exposed boundary — expected to favour fractal morphologies.

---

## Evolution  (evolution.py)

```python
run_evolution(cfg: SimConfig, n_generations: int,
              callback=default_callback, score_fn=score_plants)
    -> (final_network, score_history)
```

- All hyperparameters live in `cfg` (see `SimConfig`).  Run-control args
  (`n_generations`, `callback`, `score_fn`) are passed directly.
- Initialise `cfg.n_plants` random networks
- Each generation: run `cfg.n_steps` steps, score via `score_fn`, keep top
  `n_plants//2` exactly (elitism), replace bottom half with mutated copies
  (additive Gaussian noise, std = `cfg.sd_mut` on all weights and biases)
- Returns `(final_network, score_history)` where `score_history` is a list of
  per-generation score tensors.
- Future extensions planned: crossover, CMA-ES

### Callbacks (replaced the old `verbose` flag)
`callback(gen, scores, state, network)` is called once per generation (pass
`None` to disable).
- `default_callback` — prints per-generation best/mean/median (float-formatted,
  so it works for both integer and penalty-adjusted float scores).
- `make_alive_viewer(cmap="Greens", pause=0.001)` — returns a callback that
  live-renders the top plant's `alive` bitmap in a matplotlib window (and also
  calls `default_callback`).  Requires matplotlib.

---

## World size constraint

```python
l_world >= 2 * n_steps + 1
```
This guarantees the plant never reaches the border (plant radius ≤ n_steps).
`run_evolution` asserts this.  A runtime border-contact check is recommended
but not yet implemented.

---

## Budget clipping  (routing.py: `apply_budget_clip`)

If the sum of all `n_signal × 5` output values for a live cell exceeds
`signal_max`, all values are scaled down proportionally.
Set `signal_max = float('inf')` to disable (recommended starting point).

---

## Running the tests

```bash
python tests.py
```

All 17 tests should pass.  Tests cover: permutation table correctness, signal
routing (single cell, INTERNAL), budget clipping, division (basic, strongest-only,
race condition, occupied target), flood fills, scoring (`score_plants`,
`score_boundary_sides`, `cell_penalty` + factory), forward pass (shape and
non-negativity), evolution elitism, and a full integration smoke test.

---

## Known limitations / next steps

1. **Performance**: `route_signals` was rewritten to a shift + `scatter_add_`
   form (see Signal routing above), ~1.6× faster on CPU and ~3.3–3.7× on CUDA.
   It still loops over the 4 (now absolute) directions in Python; collapsing
   that into a single stacked scatter is a possible further optimisation.

2. **GPU**: All code is device-agnostic (pass `device=torch.device("cuda")`).
   Only `route_signals` has been benchmarked on GPU so far.

3. **Visualisation**: `make_alive_viewer` (evolution.py) live-renders the top
   plant's `alive` bitmap per generation via matplotlib.  Still missing:
   per-plant grids, signal-channel views, and saving frames/animations.

4. **Border contact check**: Should assert that `state.alive[:, 0, :].any()` etc.
   is always False; not yet implemented.

5. **Crossover**: `select_and_reproduce` only does mutation.  Crossover and
   CMA-ES are planned future extensions.

6. **Hyperparameter sensitivity**:
   - `thr_division` must be tuned relative to typical network output magnitudes
   - `sd_mut` controls exploration vs. exploitation
   - `signal_max = inf` to start; add finite value if signals diverge

7. **No gradient-based training**: This is purely evolutionary.  If gradient-based
   meta-learning is ever desired, the forward pass in `network.py` would need
   to be re-examined for differentiability (currently fine; ReLU and einsum are
   both differentiable).

---

## Suggested starting parameters for a first run

```python
from world import SimConfig
from evolution import run_evolution

cfg = SimConfig(
    n_plants     = 64,
    n_layers     = 3,
    n_signal     = 8,
    n_steps      = 20,
    l_world      = 41,          # = 2 * n_steps + 1
    thr_division = 0.5,
    signal_max   = float('inf'),
    sd_mut       = 0.05,
)   # these are also the SimConfig defaults

net, history = run_evolution(cfg, n_generations=200)
```

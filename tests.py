"""
Unit tests for the artificial plant life simulation.

Run with:  python tests.py
"""

import math
import torch
import traceback

from constants import (
    DIVISION_TRIGGER, INITIAL_SIGNAL, DIVISION_DIRECTION,
    APEX, BASE, LEFT, RIGHT, INTERNAL,
    NORTH, SOUTH, WEST, EAST,
    REL_TO_ABS, ABS_TO_REL, NEGATE_DIR, DIR_TO_DELTA,
)
from world import init_sim
from network import init_network, forward, Network
from routing import apply_budget_clip, route_signals
from division import find_division_candidates, resolve_race_conditions, apply_divisions
from scoring import flood_fill_empty, flood_fill_live, score_plants
from evolution import select_and_reproduce, step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(n_plants, l_world, n_signal):
    return init_sim(n_plants, l_world, n_signal)


def _pass(name):
    print(f"  PASS  {name}")


def _fail(name, msg):
    print(f"  FAIL  {name}: {msg}")


# ---------------------------------------------------------------------------
# 1. Permutation tables
# ---------------------------------------------------------------------------

def test_permutation_tables():
    name = "permutation_tables"
    try:
        # REL_TO_ABS and ABS_TO_REL should be mutually inverse
        for orient in range(4):
            r2a = REL_TO_ABS[orient]   # (4,)
            a2r = ABS_TO_REL[orient]   # (4,)
            # Compose: apply r2a then a2r[r2a] should give identity
            composed = a2r[r2a]
            assert (composed == torch.arange(4)).all(), \
                f"Orientation {orient}: r2a then a2r is not identity, got {composed}"
            # Reverse: apply a2r then r2a[a2r] should also give identity
            composed_rev = r2a[a2r]
            assert (composed_rev == torch.arange(4)).all(), \
                f"Orientation {orient}: a2r then r2a is not identity, got {composed_rev}"

        # NEGATE_DIR should be self-inverse
        neg2 = NEGATE_DIR[NEGATE_DIR]
        assert (neg2 == torch.arange(4)).all(), f"NEGATE_DIR not self-inverse: {neg2}"

        # Spot check: north-pointing, APEX -> NORTH
        assert REL_TO_ABS[NORTH, APEX] == NORTH
        # South-pointing, APEX -> SOUTH
        assert REL_TO_ABS[SOUTH, APEX] == SOUTH
        # West-pointing, LEFT -> SOUTH (looking west, left is south)
        assert REL_TO_ABS[WEST, LEFT] == SOUTH
        # East-pointing, RIGHT -> SOUTH (looking east, right is south)
        assert REL_TO_ABS[EAST, RIGHT] == SOUTH

        # NEGATE: NORTH <-> SOUTH, WEST <-> EAST
        assert NEGATE_DIR[NORTH] == SOUTH
        assert NEGATE_DIR[SOUTH] == NORTH
        assert NEGATE_DIR[WEST]  == EAST
        assert NEGATE_DIR[EAST]  == WEST

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 2. Routing: single cell, each orientation
# ---------------------------------------------------------------------------

def test_routing_single_cell():
    name = "routing_single_cell"
    try:
        # One plant, 5x5 world, 3 signals.
        # The routing masks dead cells to zero, so we need BOTH a sender and a
        # live receiver.  We place the sender at (cx, cx) and mark its APEX
        # neighbour as alive so the signal survives the masking step.
        l_world, n_signal = 5, 3
        cx = 2

        for orient, apex_abs in [(NORTH, NORTH), (SOUTH, SOUTH), (WEST, WEST), (EAST, EAST)]:
            state = init_sim(1, l_world, n_signal)
            state.orientation[:, cx, cx] = orient
            # Put a known value in signal 0, APEX direction
            state.output[:, cx, cx, 0, APEX] = 7.0

            # Mark the APEX neighbour as alive so the signal is not masked away
            di, dj = DIR_TO_DELTA[apex_abs].tolist()
            ni, nj = cx + di, cx + dj
            state.alive[:, ni, nj] = True
            # Neighbour points NORTH by default (set by init_sim via orientation zeros)

            new_input = route_signals(state.output, state.alive, state.orientation)

            # The arriving absolute direction is the negation of apex_abs
            arr_abs = NEGATE_DIR[apex_abs].item()
            # The relative slot at the neighbour (NORTH-pointing by default)
            arr_rel = ABS_TO_REL[NORTH, arr_abs].item()

            val = new_input[0, ni, nj, 0, arr_rel].item()
            assert abs(val - 7.0) < 1e-5, \
                f"Orient={orient}: expected 7.0 at ({ni},{nj}) rel={arr_rel}, got {val}"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 3. Routing: INTERNAL stays in place
# ---------------------------------------------------------------------------

def test_routing_internal():
    name = "routing_internal"
    try:
        l_world, n_signal = 5, 3
        cx = 2
        state = init_sim(1, l_world, n_signal)
        # Clear all output (init_sim seeds INITIAL_SIGNAL which would contaminate
        # the "no other cell received anything" assertion)
        state.output.zero_()
        state.output[:, cx, cx, 0, INTERNAL] = 5.0

        new_input = route_signals(state.output, state.alive, state.orientation)

        # INTERNAL should remain exactly at (cx, cx), signal 0
        val = new_input[0, cx, cx, 0, INTERNAL].item()
        assert abs(val - 5.0) < 1e-5, f"INTERNAL signal moved: got {val}"

        # No other position should have received anything
        new_input[0, cx, cx, 0, INTERNAL] = 0.0
        assert new_input.abs().max().item() < 1e-5, \
            "INTERNAL signal leaked to other positions"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 4. Budget clipping
# ---------------------------------------------------------------------------

def test_budget_clip():
    name = "budget_clip"
    try:
        l_world, n_signal = 5, 3
        cx = 2
        state = init_sim(1, l_world, n_signal)

        # Set all signals to 1.0 (3 signals * 5 dirs = 15 total, sum=15)
        state.output[0, cx, cx, :, :] = 1.0
        signal_max = 6.0

        clipped = apply_budget_clip(state.output, state.alive, signal_max)

        total = clipped[0, cx, cx, :, :].sum().item()
        assert abs(total - signal_max) < 1e-4, \
            f"After clip, total={total}, expected {signal_max}"

        # All values should be equal (uniform scaling)
        vals = clipped[0, cx, cx, :, :].unique()
        assert vals.shape[0] == 1, f"Values not uniform after clip: {vals}"

        # Under-budget: no change
        state2 = init_sim(1, l_world, n_signal)
        state2.output[0, cx, cx, :, :] = 0.1
        clipped2 = apply_budget_clip(state2.output, state2.alive, signal_max)
        assert (clipped2 == state2.output).all(), "Under-budget values were modified"

        # inf: no change
        state3 = init_sim(1, l_world, n_signal)
        state3.output[0, cx, cx, :, :] = 100.0
        clipped3 = apply_budget_clip(state3.output, state3.alive, float('inf'))
        assert (clipped3 == state3.output).all(), "signal_max=inf should be no-op"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 5. Division: basic case
# ---------------------------------------------------------------------------

def test_division_basic():
    name = "division_basic"
    try:
        l_world, n_signal = 7, 3
        cx = 3
        state = init_sim(1, l_world, n_signal)

        # Mother points north; set DIVISION_TRIGGER APEX signal above threshold
        thr = 0.5
        state.output[0, cx, cx, DIVISION_TRIGGER, APEX] = 1.0
        # Set all INTERNAL signals to 2.0
        state.output[0, cx, cx, :, INTERNAL] = 2.0

        # Set DIVISION_DIRECTION: strongest is APEX direction (north)
        state.output[0, cx, cx, DIVISION_DIRECTION, APEX] = 3.0

        cands = find_division_candidates(state.output, state.alive, state.orientation, thr)
        cands = resolve_race_conditions(cands, l_world, 1, state.alive.device)
        state = apply_divisions(state, cands)

        # Daughter should be at (cx-1, cx) (north of mother)
        assert state.alive[0, cx-1, cx].item(), "Daughter cell not alive"

        # Mother internal signals should be halved
        for s in range(n_signal):
            val = state.output[0, cx, cx, s, INTERNAL].item()
            assert abs(val - 1.0) < 1e-5, \
                f"Mother signal {s} INTERNAL not halved: {val}"

        # Daughter internal signals = mother post-halved = 1.0
        for s in range(n_signal):
            val = state.output[0, cx-1, cx, s, INTERNAL].item()
            assert abs(val - 1.0) < 1e-5, \
                f"Daughter signal {s} INTERNAL wrong: {val}"

        # Daughter directional outputs should be zero
        assert state.output[0, cx-1, cx, :, :4].abs().max().item() < 1e-5, \
            "Daughter directional outputs not zero"

        # Daughter orientation: DIVISION_DIRECTION argmax was APEX (north) -> NORTH
        assert state.orientation[0, cx-1, cx].item() == NORTH, \
            f"Daughter orientation wrong: {state.orientation[0, cx-1, cx].item()}"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 6. Division: only strongest direction fires
# ---------------------------------------------------------------------------

def test_division_strongest_only():
    name = "division_strongest_only"
    try:
        l_world, n_signal = 7, 3
        cx = 3
        state = init_sim(1, l_world, n_signal)
        thr = 0.5

        # Two directions above threshold; SOUTH is stronger
        state.output[0, cx, cx, DIVISION_TRIGGER, APEX]  = 0.8
        state.output[0, cx, cx, DIVISION_TRIGGER, BASE]  = 1.5   # SOUTH = stronger

        cands = find_division_candidates(state.output, state.alive, state.orientation, thr)
        cands = resolve_race_conditions(cands, l_world, 1, state.alive.device)
        state = apply_divisions(state, cands)

        # Only south neighbour (cx+1, cx) should be alive
        assert state.alive[0, cx+1, cx].item(),   "South daughter not alive"
        assert not state.alive[0, cx-1, cx].item(), "North daughter should not exist"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 7. Division: race condition
# ---------------------------------------------------------------------------

def test_division_race_condition():
    name = "division_race_condition"
    try:
        l_world, n_signal = 9, 3
        cx = 4
        state = init_sim(1, l_world, n_signal)
        thr = 0.5

        # Cell A at (cx, cx) points north, APEX signal = 2.0 -> targets (cx-1, cx)
        # Cell B at (cx-2, cx) points south, APEX signal = 3.0 -> targets (cx-1, cx)
        # Both target (cx-1, cx); B wins (higher signal)

        state.alive[0, cx-2, cx] = True
        state.orientation[0, cx-2, cx] = SOUTH

        state.output[0, cx,   cx, DIVISION_TRIGGER, APEX] = 2.0  # A
        state.output[0, cx-2, cx, DIVISION_TRIGGER, APEX] = 3.0  # B (stronger)
        # B: APEX for SOUTH-pointing cell is direction SOUTH, but the absolute
        #    target for B's APEX is SOUTH direction -> (cx-2+1, cx) = (cx-1, cx). Correct.

        cands = find_division_candidates(state.output, state.alive, state.orientation, thr)
        cands = resolve_race_conditions(cands, l_world, 1, state.alive.device)

        # Should have exactly 1 surviving candidate
        assert cands['p'].numel() == 1, \
            f"Expected 1 surviving candidate, got {cands['p'].numel()}"

        # The winner should be cell B (value=3.0)
        assert abs(cands['value'][0].item() - 3.0) < 1e-5, \
            f"Wrong winner: value={cands['value'][0].item()}"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 8. Division: occupied target -> no division
# ---------------------------------------------------------------------------

def test_division_occupied():
    name = "division_occupied"
    try:
        l_world, n_signal = 7, 3
        cx = 3
        state = init_sim(1, l_world, n_signal)
        thr = 0.5

        # Put a live cell north of the seed
        state.alive[0, cx-1, cx] = True
        # Seed tries to divide north (above threshold)
        state.output[0, cx, cx, DIVISION_TRIGGER, APEX] = 2.0

        cands = find_division_candidates(state.output, state.alive, state.orientation, thr)
        assert cands['p'].numel() == 0, \
            f"Division candidate found even though target is occupied"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 9. Flood fill: empty cells from border
# ---------------------------------------------------------------------------

def test_flood_fill_empty():
    name = "flood_fill_empty"
    try:
        l_world = 7
        alive = torch.zeros(1, l_world, l_world, dtype=torch.bool)

        # Place a horizontal bar of live cells across the middle column,
        # dividing the grid into left and right halves (cells (0..6, 3))
        alive[0, :, 3] = True

        result = flood_fill_empty(alive, l_world)

        # Right half (cols 4-6) and left half (cols 0-2) should both be reachable
        # from border via empty cells (the bar doesn't seal off either side since
        # border rows 0 and 6 are empty above/below the bar)
        # Cols 0-2 are directly on the border (col 0)
        assert result[0, 1, 0].item(), "Left of bar should be reachable"
        assert result[0, 1, 6].item(), "Right of bar should be reachable"

        # Live cells themselves should NOT be in the empty fill
        assert not result[0, 1, 3].item(), "Live cell should not be in empty fill"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 10. Flood fill: live connected component
# ---------------------------------------------------------------------------

def test_flood_fill_live():
    name = "flood_fill_live"
    try:
        l_world = 9
        cx = 4
        alive = torch.zeros(1, l_world, l_world, dtype=torch.bool)

        # Seed cell alive
        alive[0, cx, cx] = True
        # Connected neighbour
        alive[0, cx+1, cx] = True
        # Disconnected cell (not touching any of the above)
        alive[0, 1, 1] = True

        result = flood_fill_live(alive, l_world)

        assert result[0, cx, cx].item(),   "Seed cell should be in component"
        assert result[0, cx+1, cx].item(), "Connected neighbour should be in component"
        assert not result[0, 1, 1].item(), "Disconnected cell should not be in component"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 11. Scoring: end-to-end on a known grid
# ---------------------------------------------------------------------------

def test_scoring_end_to_end():
    name = "scoring_end_to_end"
    try:
        # Use a small world where we can reason manually.
        # 7x7 world, seed at (3,3).
        # Place live cells at (3,3) (seed) and (3,4).
        # (3,3) is surrounded by (3,2)=empty, (3,4)=live, (2,3)=empty, (4,3)=empty
        #   -> it borders empty space reachable from border -> should score
        # (3,4) is surrounded by (3,3)=live, (3,5)=empty, (2,4)=empty, (4,4)=empty
        #   -> it borders empty space reachable from border -> should score
        # Both are seed-connected.
        # Expected score = 2.
        l_world = 7
        cx = 3
        alive = torch.zeros(1, l_world, l_world, dtype=torch.bool)
        alive[0, cx, cx]   = True
        alive[0, cx, cx+1] = True

        scores = score_plants(alive, l_world)
        assert scores[0].item() == 2, f"Expected score 2, got {scores[0].item()}"

        # A single isolated seed should score 1
        alive2 = torch.zeros(1, l_world, l_world, dtype=torch.bool)
        alive2[0, cx, cx] = True
        scores2 = score_plants(alive2, l_world)
        assert scores2[0].item() == 1, f"Single seed: expected score 1, got {scores2[0].item()}"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 12. Forward pass: shape
# ---------------------------------------------------------------------------

def test_forward_pass_shape():
    name = "forward_pass_shape"
    try:
        n_plants, n_layers, n_signal = 4, 2, 3
        l_world = 11
        net = init_network(n_plants, n_layers, n_signal)
        x = torch.randn(n_plants, l_world, l_world, n_signal, 5)
        out = forward(net, x)
        expected = (n_plants, l_world, l_world, n_signal, 5)
        assert out.shape == expected, f"Shape mismatch: {out.shape} vs {expected}"
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 13. Forward pass: non-negative outputs
# ---------------------------------------------------------------------------

def test_forward_pass_nonneg():
    name = "forward_pass_nonneg"
    try:
        n_plants, n_layers, n_signal = 4, 3, 4
        l_world = 7
        net = init_network(n_plants, n_layers, n_signal)
        # Use negative inputs to stress-test ReLU
        x = torch.randn(n_plants, l_world, l_world, n_signal, 5) * 5 - 3
        out = forward(net, x)
        assert (out >= 0).all(), f"Negative output found: min={out.min().item()}"
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 14. Evolution: elitism
# ---------------------------------------------------------------------------

def test_evolution_elitism():
    name = "evolution_elitism"
    try:
        n_plants, n_layers, n_signal = 6, 2, 3
        net = init_network(n_plants, n_layers, n_signal)

        # Assign scores: plants 0,1,2 are best
        scores = torch.tensor([10, 9, 8, 1, 1, 1], dtype=torch.long)

        new_net = select_and_reproduce(net, scores, sd_mut=0.1)

        # Top 3 plants (plants 0,1,2 by score rank) should appear unchanged
        # in the first 3 rows of the new network
        n_keep = n_plants // 2  # 3
        order  = torch.argsort(scores, descending=True)
        top    = order[:n_keep]

        for rank, orig_idx in enumerate(top):
            orig_w = net.weights[orig_idx]
            new_w  = new_net.weights[rank]
            assert torch.equal(orig_w, new_w), \
                f"Elite rank {rank} (original plant {orig_idx}) was modified"

        # Mutated half should differ from their source
        all_same = True
        for rank in range(n_keep, n_plants):
            src_rank = rank - n_keep
            if not torch.equal(new_net.weights[rank], new_net.weights[src_rank]):
                all_same = False
                break
        assert not all_same, "Mutated copies appear identical to elites (sd_mut too small?)"

        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# 15. Integration: a few steps without crashing
# ---------------------------------------------------------------------------

def test_integration_smoke():
    name = "integration_smoke"
    try:
        n_plants, n_layers, n_signal = 4, 2, 3
        n_steps, l_world = 5, 15
        thr_division = 0.1
        signal_max   = float('inf')

        net   = init_network(n_plants, n_layers, n_signal)
        state = init_sim(n_plants, l_world, n_signal)

        for _ in range(n_steps):
            state = step(state, net, thr_division, signal_max)

        scores = score_plants(state.alive, l_world)
        assert scores.shape == (n_plants,), f"Scores shape wrong: {scores.shape}"
        assert (scores >= 0).all(), "Negative scores"

        _pass(name)
    except Exception as e:
        _fail(name, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_permutation_tables,
        test_routing_single_cell,
        test_routing_internal,
        test_budget_clip,
        test_division_basic,
        test_division_strongest_only,
        test_division_race_condition,
        test_division_occupied,
        test_flood_fill_empty,
        test_flood_fill_live,
        test_scoring_end_to_end,
        test_forward_pass_shape,
        test_forward_pass_nonneg,
        test_evolution_elitism,
        test_integration_smoke,
    ]

    print(f"\nRunning {len(tests)} tests\n" + "-" * 40)
    for t in tests:
        try:
            t()
        except Exception as e:
            _fail(t.__name__, f"Uncaught exception: {e}\n{traceback.format_exc()}")
    print("-" * 40)

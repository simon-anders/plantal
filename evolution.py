"""
Evolution loop: step execution, generation runner, and evolutionary selection.
"""

import math
import torch
from torch import Tensor

from world import SimState, init_sim
from network import Network, init_network, forward
from routing import apply_budget_clip, route_signals
from division import find_division_candidates, resolve_race_conditions, apply_divisions
from scoring import score_plants


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def step(
    state:        SimState,
    network:      Network,
    thr_division: float,
    signal_max:   float,
) -> SimState:
    """Execute one simulation time step in-place.

    Order:
      1. Forward pass (all layers, ReLU after each)
      2. Budget clipping
      3. Division processing
      4. Signal routing -> new input (stored back into state.output)
      5. Dead-cell masking (handled inside route_signals)

    Parameters
    ----------
    state        : SimState  (modified in-place)
    network      : Network
    thr_division : float     division threshold
    signal_max   : float     use math.inf to disable budget clipping

    Returns
    -------
    state (same object)
    """
    # 1 & 2: forward pass then budget clip
    output = forward(network, state.output)
    output = apply_budget_clip(output, state.alive, signal_max)
    state.output = output

    # 3: division
    candidates = find_division_candidates(
        state.output, state.alive, state.orientation, thr_division
    )
    candidates = resolve_race_conditions(
        candidates, state.alive.shape[1], state.alive.shape[0], state.alive.device
    )
    state = apply_divisions(state, candidates)

    # 4 & 5: routing and masking
    state.output = route_signals(state.output, state.alive, state.orientation)

    return state


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------

def run_generation(
    network:      Network,
    n_steps:      int,
    l_world:      int,
    n_signal:     int,
    thr_division: float,
    signal_max:   float,
    device:       torch.device,
) -> tuple[Tensor, SimState]:
    """Run one generation (n_steps steps) and return scores and final state.

    Parameters
    ----------
    network      : Network   (n_plants, ...)
    n_steps      : int
    l_world      : int
    n_signal     : int
    thr_division : float
    signal_max   : float
    device       : torch.device

    Returns
    -------
    (scores, final_state)
      scores      : LongTensor (n_plants,)
      final_state : SimState
    """
    n_plants = network.weights.shape[0]
    state = init_sim(n_plants, l_world, n_signal, device)

    for _ in range(n_steps):
        state = step(state, network, thr_division, signal_max)

    scores = score_plants(state.alive, l_world)
    return scores, state


# ---------------------------------------------------------------------------
# Selection and reproduction
# ---------------------------------------------------------------------------

def select_and_reproduce(
    network: Network,
    scores:  Tensor,
    sd_mut:  float,
) -> Network:
    """Keep top half of plants (by score) and replace the bottom half with
    mutated copies.  The top half is preserved exactly (elitism).

    Parameters
    ----------
    network : Network  (n_plants, ...)
    scores  : LongTensor (n_plants,)
    sd_mut  : float     std of additive Gaussian mutation noise

    Returns
    -------
    New Network with the same shape.
    """
    n_plants = network.weights.shape[0]
    n_keep   = n_plants // 2

    # Rank by score (descending)
    order   = torch.argsort(scores, descending=True)
    top_idx = order[:n_keep]   # indices of the best plants

    # Replicate top plants to fill both halves
    new_weights = network.weights[top_idx].repeat(2, 1, 1, 1, 1, 1)[:n_plants]
    new_biases  = network.biases[top_idx].repeat(2, 1, 1, 1)[:n_plants]

    # The first n_keep rows are the preserved elites (no mutation)
    # The second n_keep rows get mutation noise added
    n_mutated = n_plants - n_keep

    noise_w = torch.randn_like(new_weights[n_keep:]) * sd_mut
    noise_b = torch.randn_like(new_biases[n_keep:])  * sd_mut

    new_weights[n_keep:] = new_weights[n_keep:] + noise_w
    new_biases[n_keep:]  = new_biases[n_keep:]  + noise_b

    return Network(weights=new_weights, biases=new_biases)


# ---------------------------------------------------------------------------
# Outer evolution loop
# ---------------------------------------------------------------------------

def run_evolution(
    n_generations: int,
    n_plants:      int,
    l_world:       int,
    n_signal:      int,
    n_layers:      int,
    n_steps:       int,
    thr_division:  float,
    signal_max:    float,
    sd_mut:        float,
    device:        torch.device = torch.device("cpu"),
    verbose:       bool = True,
) -> tuple[Network, list[Tensor]]:
    """Full evolutionary run.

    Returns
    -------
    (final_network, score_history)
      final_network  : Network after the last selection step
      score_history  : list of LongTensor (n_plants,), one per generation
    """
    assert l_world >= 2 * n_steps + 1, (
        f"l_world={l_world} is too small; need >= {2 * n_steps + 1}"
    )

    network = init_network(n_plants, n_layers, n_signal, device)
    score_history = []

    for gen in range(n_generations):
        scores, _ = run_generation(
            network, n_steps, l_world, n_signal, thr_division, signal_max, device
        )
        score_history.append(scores.clone())

        if verbose:
            print(
                f"Gen {gen+1:4d}  "
                f"best={scores.max().item():6d}  "
                f"mean={scores.float().mean().item():7.1f}  "
                f"median={scores.median().item():6d}"
            )

        network = select_and_reproduce(network, scores, sd_mut)

    return network, score_history

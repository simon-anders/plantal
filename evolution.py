"""
Evolution loop: step execution, generation runner, and evolutionary selection.
"""

import math
from typing import Callable
import torch
from torch import Tensor

from world import SimConfig, SimState, init_sim
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
    network:  Network,
    cfg:      SimConfig,
    score_fn: Callable[[Tensor, int], Tensor] = score_plants,
) -> tuple[Tensor, SimState]:
    """Run one generation (cfg.n_steps steps) and return scores and final state.

    Parameters
    ----------
    network  : Network   (n_plants, ...)
    cfg      : SimConfig
    score_fn : callable  fitness function ``score_fn(alive, l_world) -> scores``;
               defaults to `score_plants`.

    Returns
    -------
    (scores, final_state)
      scores      : LongTensor (n_plants,)
      final_state : SimState
    """
    n_plants = network.weights.shape[0]
    state = init_sim(n_plants, cfg.l_world, cfg.n_signal, cfg.device)

    for _ in range(cfg.n_steps):
        state = step(state, network, cfg.thr_division, cfg.signal_max)

    scores = score_fn(state.alive, cfg.l_world)
    return scores, state


# ---------------------------------------------------------------------------
# Selection and reproduction
# ---------------------------------------------------------------------------

def select_and_reproduce(
    network: Network,
    scores:  Tensor,
    cfg:     SimConfig,
) -> Network:
    """Keep top half of plants (by score) and replace the bottom half with
    mutated copies.  The top half is preserved exactly (elitism).

    Parameters
    ----------
    network : Network  (n_plants, ...)
    scores  : LongTensor (n_plants,)
    cfg     : SimConfig  (uses cfg.sd_mut for mutation noise)

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

    noise_w = torch.randn_like(new_weights[n_keep:]) * cfg.sd_mut
    noise_b = torch.randn_like(new_biases[n_keep:])  * cfg.sd_mut

    new_weights[n_keep:] = new_weights[n_keep:] + noise_w
    new_biases[n_keep:]  = new_biases[n_keep:]  + noise_b

    return Network(weights=new_weights, biases=new_biases)


# ---------------------------------------------------------------------------
# Outer evolution loop
# ---------------------------------------------------------------------------

def default_callback(gen: int, scores: Tensor, state: SimState, network: Network) -> None:
    """Default per-generation callback: print summary statistics.

    Reproduces the old `verbose=True` output.
    """
    # Scores may be integer (default scorers) or float (e.g. with a size
    # penalty), so format numerically rather than with an integer code.
    print(
        f"Gen {gen+1:4d}  "
        f"best={scores.max().item():8.2f}  "
        f"mean={scores.float().mean().item():8.2f}  "
        f"median={scores.float().median().item():8.2f}"
    )


def make_alive_viewer(
    cmap:  str = "Greens",
    pause: float = 0.001,
):
    """Open a matplotlib window and return a per-generation callback that
    displays the final alive matrix of the top-scoring plant, updating the
    same window in place each generation.

    The returned closure also calls `default_callback` at the end, so the
    text summary is printed alongside the graphical update.

    Parameters
    ----------
    cmap  : str    matplotlib colormap for the alive bitmap
    pause : float  seconds passed to plt.pause to let the GUI redraw

    Returns
    -------
    callback(gen, scores, state, network) -> None
    """
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow([[0]], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()

    def callback(gen: int, scores: Tensor, state: SimState, network: Network) -> None:
        best = int(scores.argmax())
        im.set_data(state.alive[best].cpu().numpy())
        ax.set_title(f"Gen {gen+1}  plant {best}  score {int(scores[best])}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause)

        default_callback(gen, scores, state, network)

    return callback


def run_evolution(
    cfg:           SimConfig,
    n_generations: int,
    callback:      Callable[[int, Tensor, SimState, Network], None] | None = default_callback,
    score_fn:      Callable[[Tensor, int], Tensor] = score_plants,
) -> tuple[Network, list[Tensor]]:
    """Full evolutionary run.

    Parameters
    ----------
    cfg : SimConfig
        Simulation and evolution hyperparameters.  (Validation of l_world and
        n_signal happens at SimConfig construction.)
    n_generations : int
        Number of generations to run.
    callback : callable or None
        Called once per generation as ``callback(gen, scores, state, network)``,
        where ``gen`` is the zero-based generation index, ``scores`` is the
        generation's score tensor, ``state`` is the final SimState, and
        ``network`` is the network that produced them (before
        selection/reproduction).  Pass ``None`` to disable.
        Defaults to `default_callback`, which prints summary statistics.
    score_fn : callable
        Fitness function ``score_fn(alive, l_world) -> scores``, evaluated on
        the final state of each generation.  Defaults to `score_plants`.

    Returns
    -------
    (final_network, score_history)
      final_network  : Network after the last selection step
      score_history  : list of LongTensor (n_plants,), one per generation
    """
    network = init_network(cfg.n_plants, cfg.n_layers, cfg.n_signal, cfg.device)
    score_history = []

    for gen in range(n_generations):
        scores, state = run_generation(network, cfg, score_fn)
        score_history.append(scores.clone())

        if callback is not None:
            callback(gen, scores, state, network)

        network = select_and_reproduce(network, scores, cfg)

    return network, score_history

"""
Simulation state dataclass and initialisation.
"""

import math
from dataclasses import dataclass, field
import torch
from torch import Tensor

from constants import (
    INITIAL_SIGNAL, INTERNAL, NORTH
)


@dataclass(frozen=True)
class SimConfig:
    """Static simulation and evolution hyperparameters.

    A frozen, single source of truth threaded through the orchestration layer
    (run_evolution / run_generation / select_and_reproduce), which unpacks the
    individual primitives needed by the leaf functions.  Run-control settings
    (n_generations, callback) are deliberately kept out of here.

    Attributes
    ----------
    n_plants     : int    number of parallel plants/worlds
    l_world      : int    grid side length; must be >= 2 * n_steps + 1
    n_signal     : int    number of signal channels; must be >= 3
    n_layers     : int    number of dense layers in each plant's network
    n_steps      : int    simulation steps per generation
    thr_division : float  division trigger threshold
    signal_max   : float  per-cell signal budget; math.inf disables clipping
    sd_mut       : float  std of additive Gaussian mutation noise
    device       : torch.device
    """
    n_plants:     int   = 64
    l_world:      int   = 41
    n_signal:     int   = 8
    n_layers:     int   = 3
    n_steps:      int   = 20
    thr_division: float = 0.5
    signal_max:   float = math.inf
    sd_mut:       float = 0.05
    device:       torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        assert self.n_signal >= 3, "n_signal must be at least 3"
        assert self.l_world >= 2 * self.n_steps + 1, (
            f"l_world={self.l_world} is too small; need >= {2 * self.n_steps + 1}"
        )


@dataclass
class SimState:
    """All mutable state for n_plants parallel worlds.

    Attributes
    ----------
    alive : BoolTensor (n_plants, l_world, l_world)
        True where a live cell exists.
    orientation : LongTensor (n_plants, l_world, l_world)
        Absolute direction index of each cell's apex (NORTH/SOUTH/WEST/EAST).
        Values at dead cells are meaningless.
    output : FloatTensor (n_plants, l_world, l_world, n_signal, 5)
        The output tensor produced by the forward pass (and modified in-place
        by division processing).  After routing it becomes the next input.
    """
    alive:       Tensor   # bool  (n_plants, l_world, l_world)
    orientation: Tensor   # long  (n_plants, l_world, l_world)
    output:      Tensor   # float (n_plants, l_world, l_world, n_signal, 5)


def init_sim(
    n_plants: int,
    l_world:  int,
    n_signal: int,
    device:   torch.device = torch.device("cpu"),
) -> SimState:
    """Create the initial simulation state.

    All worlds start with a single alive seed cell at (l_world//2, l_world//2),
    orientation NORTH, and INITIAL_SIGNAL set to 1 in the INTERNAL slot.

    Parameters
    ----------
    n_plants : int
    l_world  : int   must satisfy l_world >= 2 * n_steps + 1
    n_signal : int   must be >= 3
    device   : torch.device
    """
    assert n_signal >= 3, "n_signal must be at least 3"

    cx = l_world // 2  # seed row
    cy = l_world // 2  # seed col

    alive = torch.zeros(n_plants, l_world, l_world, dtype=torch.bool, device=device)
    alive[:, cx, cy] = True

    orientation = torch.zeros(n_plants, l_world, l_world, dtype=torch.long, device=device)
    orientation[:, cx, cy] = NORTH

    # output tensor also serves as the initial input tensor (before the first
    # forward pass it is the cell's initial state).
    output = torch.zeros(n_plants, l_world, l_world, n_signal, 5,
                         dtype=torch.float32, device=device)
    output[:, cx, cy, INITIAL_SIGNAL, INTERNAL] = 1.0

    return SimState(alive=alive, orientation=orientation, output=output)

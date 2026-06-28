"""
Simulation state dataclass and initialisation.
"""

from dataclasses import dataclass
import torch
from torch import Tensor

from constants import (
    INITIAL_SIGNAL, INTERNAL, NORTH
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

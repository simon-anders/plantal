"""
Signal routing: propagates directional outputs from each cell to the
appropriate input slot of the appropriate neighbour.

Also handles budget clipping.
"""

import torch
from torch import Tensor

from constants import (
    INTERNAL,
    ABS_TO_REL, NEGATE_DIR, DIR_TO_DELTA,
)


def apply_budget_clip(
    output: Tensor,       # (n_plants, l_world, l_world, n_signal, 5)
    alive:  Tensor,       # (n_plants, l_world, l_world)  bool
    signal_max: float,
) -> Tensor:
    """Scale down each live cell's output if its total signal sum exceeds signal_max.

    The sum is taken over all n_signal * 5 values for each cell.
    Dead cells are untouched (they are already zero).

    Parameters
    ----------
    output     : float tensor (n_plants, l_world, l_world, n_signal, 5)
    alive      : bool  tensor (n_plants, l_world, l_world)
    signal_max : float   use math.inf to disable

    Returns
    -------
    Clipped output tensor (same shape, in-place modification avoided).
    """
    if signal_max == float('inf'):
        return output

    # Sum over signal and direction axes -> (n_plants, l_world, l_world)
    total = output.sum(dim=(-2, -1))  # (n_plants, l_world, l_world)

    # Scale factor: 1.0 where under budget, signal_max/total where over
    scale = torch.where(total > signal_max, signal_max / total.clamp(min=1e-12),
                        torch.ones_like(total))

    # Only apply to live cells
    scale = torch.where(alive, scale, torch.ones_like(scale))

    # Broadcast scale over (n_signal, 5) dims
    return output * scale.unsqueeze(-1).unsqueeze(-1)


def _shift2d(x: Tensor, di: int, dj: int) -> Tensor:
    """Shift x on grid axes (1, 2) so out[:, i, j] = x[:, i - di, j - dj].

    Border cells that would pull from outside the grid are zero-filled (no
    wrap-around).  Only single-step shifts (|di|, |dj| <= 1) are used here.
    """
    out = torch.roll(x, shifts=(di, dj), dims=(1, 2))
    if di == 1:
        out[:, 0, :, :] = 0
    elif di == -1:
        out[:, -1, :, :] = 0
    if dj == 1:
        out[:, :, 0, :] = 0
    elif dj == -1:
        out[:, :, -1, :] = 0
    return out


def route_signals(
    output:      Tensor,   # (n_plants, l_world, l_world, n_signal, 5)
    alive:       Tensor,   # (n_plants, l_world, l_world)  bool
    orientation: Tensor,   # (n_plants, l_world, l_world)  long
) -> Tensor:
    """Build the new input tensor by routing directional outputs to neighbours.

    Reformulated by ABSOLUTE direction of travel a in {NORTH, SOUTH, WEST, EAST}.
    For a fixed a every cell performs the same grid shift, so routing is a dense
    spatial shift plus a scatter over the small relative-slot axis -- no
    per-element boolean-masked index_put_.  For each a:
      1. The outgoing signal a cell sends in direction a lives in relative slot
         ABS_TO_REL[sender_orientation, a] (the inverse of REL_TO_ABS); gather it.
      2. Zero dead senders, then shift the whole field by DIR_TO_DELTA[a].
      3. The signal arrives at face a_neg = NEGATE_DIR[a]; at each destination it
         lands in relative slot ABS_TO_REL[dest_orientation, a_neg].
      4. scatter_add_ into that slot.

    INTERNAL signals (index 4) stay in place.  Dead-cell positions are zeroed at
    the end, so aliveness of the destination need not be checked during routing.

    Parameters
    ----------
    output      : float (n_plants, l_world, l_world, n_signal, 5)
    alive       : bool  (n_plants, l_world, l_world)
    orientation : long  (n_plants, l_world, l_world)

    Returns
    -------
    new_input : float (n_plants, l_world, l_world, n_signal, 5)
        Signals are in the correct relative-direction slots for each cell.
        Dead cell positions are zeroed.
    """
    device = output.device
    n_signal = output.shape[-2]

    # Move look-up tables to the right device
    abs_to_rel = ABS_TO_REL.to(device)   # (4, 4)
    negate_dir = NEGATE_DIR.to(device)   # (4,)
    dir_delta  = DIR_TO_DELTA.to(device) # (4, 2)

    new_input = torch.zeros_like(output)

    # --- INTERNAL signals: copy in-place, only for live cells ---
    new_input[..., INTERNAL] = output[..., INTERNAL] * alive.unsqueeze(-1)

    alive_f = alive.unsqueeze(-1)   # (n_plants, l_world, l_world, 1)

    # --- Directional signals, one absolute direction of travel at a time ---
    for a in range(4):   # NORTH, SOUTH, WEST, EAST
        # 1. Gather the signal each cell sends in absolute direction a.
        rel_s = abs_to_rel[orientation, a]                        # (P, L, L)
        gather_idx = rel_s.unsqueeze(-1).unsqueeze(-1).expand(
            *orientation.shape, n_signal, 1)                      # (P, L, L, S, 1)
        sig = output.gather(-1, gather_idx).squeeze(-1)           # (P, L, L, S)
        sig = sig * alive_f                                       # dead senders send nothing

        # 2. Shift the whole field one step in direction a (zero-filled border).
        di = int(dir_delta[a, 0]); dj = int(dir_delta[a, 1])
        shifted = _shift2d(sig, di, dj)                           # (P, L, L, S)

        # 3. Destination relative slot for the arriving (negated) face.
        a_neg = int(negate_dir[a])
        dest_slot = abs_to_rel[orientation, a_neg]                # (P, L, L)

        # 4. scatter_add_ into that slot of new_input's last axis.
        scatter_idx = dest_slot.unsqueeze(-1).expand(
            *orientation.shape, n_signal).unsqueeze(-1)           # (P, L, L, S, 1)
        new_input.scatter_add_(-1, scatter_idx, shifted.unsqueeze(-1))

    # Zero out dead cells (destinations were never masked during the scatter).
    new_input[~alive] = 0.0

    return new_input

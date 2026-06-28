"""
Signal routing: propagates directional outputs from each cell to the
appropriate input slot of the appropriate neighbour.

Also handles budget clipping.
"""

import torch
from torch import Tensor

from constants import (
    INTERNAL,
    REL_TO_ABS, ABS_TO_REL, NEGATE_DIR, DIR_TO_DELTA,
    NORTH, SOUTH, WEST, EAST,
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


def route_signals(
    output:      Tensor,   # (n_plants, l_world, l_world, n_signal, 5)
    alive:       Tensor,   # (n_plants, l_world, l_world)  bool
    orientation: Tensor,   # (n_plants, l_world, l_world)  long
) -> Tensor:
    """Build the new input tensor by routing directional outputs to neighbours.

    For each live cell and each relative direction r in {APEX, BASE, LEFT, RIGHT}:
      1. Convert r -> absolute direction a using REL_TO_ABS[cell_orientation, r]
      2. Negate a to get the face at which the signal arrives: a_neg = NEGATE_DIR[a]
      3. Shift the signal to the neighbouring grid square given by DIR_TO_DELTA[a]
      4. At the neighbour, convert a_neg -> relative input slot using
         ABS_TO_REL[neighbour_orientation, a_neg]
      5. Accumulate into the new input tensor

    INTERNAL signals (index 4) stay in place.

    Dead cell outputs are zero (enforced by masking after routing), so no
    special handling is needed for them during the scatter.

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
    n_plants, l_world, _, n_signal, _ = output.shape
    device = output.device

    # Move look-up tables to the right device
    rel_to_abs = REL_TO_ABS.to(device)   # (4, 4)
    abs_to_rel = ABS_TO_REL.to(device)   # (4, 4)
    negate_dir = NEGATE_DIR.to(device)   # (4,)
    dir_delta  = DIR_TO_DELTA.to(device) # (4, 2)

    new_input = torch.zeros_like(output)

    # --- INTERNAL signals: copy in-place, only for live cells ---
    # output[..., INTERNAL] stays at same grid position
    new_input[..., INTERNAL] = output[..., INTERNAL] * alive.unsqueeze(-1)

    # --- Directional signals (rel_dir in 0..3) ---
    # We iterate over the 4 relative directions; each is a vectorised scatter.
    for rel_dir in range(4):   # APEX, BASE, LEFT, RIGHT
        # abs_dir for each cell: (n_plants, l_world, l_world)
        abs_dir = rel_to_abs[orientation, rel_dir]   # broadcast lookup

        # Grid shift for this absolute direction
        # di, dj: (n_plants, l_world, l_world)
        di = dir_delta[abs_dir, 0]
        dj = dir_delta[abs_dir, 1]

        # Build destination indices
        rows = torch.arange(l_world, device=device).view(1, l_world, 1).expand(n_plants, l_world, l_world)
        cols = torch.arange(l_world, device=device).view(1, 1, l_world).expand(n_plants, l_world, l_world)

        dest_i = (rows + di).clamp(0, l_world - 1)  # (n_plants, l_world, l_world)
        dest_j = (cols + dj).clamp(0, l_world - 1)

        # Mask: valid only if source is alive AND destination is within bounds
        in_bounds = (rows + di >= 0) & (rows + di < l_world) & \
                    (cols + dj >= 0) & (cols + dj < l_world)
        valid = alive & in_bounds  # (n_plants, l_world, l_world)

        # Signals to send: (n_plants, l_world, l_world, n_signal)
        signals = output[..., rel_dir]   # (n_plants, l_world, l_world, n_signal)

        # Arriving absolute direction (negated)
        abs_dir_neg = negate_dir[abs_dir]   # (n_plants, l_world, l_world)

        # Orientation of destination cells
        p_idx = torch.arange(n_plants, device=device).view(n_plants, 1, 1).expand_as(dest_i)
        dest_orient = orientation[p_idx, dest_i, dest_j]   # (n_plants, l_world, l_world)

        # Relative input slot at the destination
        dest_rel = abs_to_rel[dest_orient, abs_dir_neg]   # (n_plants, l_world, l_world)

        # Scatter into new_input using index_put_ with accumulate
        # We need to scatter (n_plants, l_world, l_world, n_signal) values
        # into new_input[p, dest_i, dest_j, :, dest_rel]
        # Flatten plant/source dims for indexing
        p_flat    = p_idx[valid]         # (N_valid,)
        di_flat   = dest_i[valid]        # (N_valid,)
        dj_flat   = dest_j[valid]        # (N_valid,)
        dr_flat   = dest_rel[valid]      # (N_valid,)
        sig_flat  = signals[valid]       # (N_valid, n_signal)

        # Expand indices over the n_signal axis
        n_valid = p_flat.shape[0]
        p_exp  = p_flat.unsqueeze(1).expand(n_valid, n_signal)
        di_exp = di_flat.unsqueeze(1).expand(n_valid, n_signal)
        dj_exp = dj_flat.unsqueeze(1).expand(n_valid, n_signal)
        s_exp  = torch.arange(n_signal, device=device).unsqueeze(0).expand(n_valid, n_signal)
        dr_exp = dr_flat.unsqueeze(1).expand(n_valid, n_signal)

        new_input.index_put_(
            (p_exp.reshape(-1), di_exp.reshape(-1), dj_exp.reshape(-1),
             s_exp.reshape(-1), dr_exp.reshape(-1)),
            sig_flat.reshape(-1),
            accumulate=True,
        )

    # Zero out dead cells (in case anything leaked via clamped out-of-bounds indices)
    new_input[~alive] = 0.0

    return new_input

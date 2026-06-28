"""
Cell division logic.

Processing order (within a step, after budget clipping):
  1. find_division_candidates  -- identify all (plant, i, j, abs_dir, value) tuples
  2. resolve_race_conditions   -- drop conflicts, keep highest signal
  3. apply_divisions           -- batch-update alive, orientation, output
"""

import torch
from torch import Tensor
from world import SimState
from constants import (
    DIVISION_TRIGGER, DIVISION_DIRECTION, INTERNAL,
    REL_TO_ABS, ABS_TO_REL, DIR_TO_DELTA, NEGATE_DIR,
    NORTH,
)


def find_division_candidates(
    output:        Tensor,   # (n_plants, l_world, l_world, n_signal, 5)
    alive:         Tensor,   # (n_plants, l_world, l_world)  bool
    orientation:   Tensor,   # (n_plants, l_world, l_world)  long
    thr_division:  float,
) -> dict:
    """Find the strongest valid division signal for each live cell.

    A directional output of DIVISION_TRIGGER exceeds thr_division and the
    target square is empty.  Per cell we only follow the strongest direction.

    Returns
    -------
    dict with keys:
      'p'     : LongTensor (N,)  plant index
      'i'     : LongTensor (N,)  source row
      'j'     : LongTensor (N,)  source col
      'abs_d' : LongTensor (N,)  absolute direction of division
      'value' : FloatTensor (N,) signal value that triggered division
    """
    device = output.device
    n_plants, l_world, _ = alive.shape

    rel_to_abs = REL_TO_ABS.to(device)   # (4, 4)
    dir_delta  = DIR_TO_DELTA.to(device) # (4, 2)

    # div_signals: (n_plants, l_world, l_world, 4)
    # The four directional outputs of DIVISION_TRIGGER (exclude INTERNAL)
    div_signals = output[..., DIVISION_TRIGGER, :4]

    # Convert relative direction axis to absolute direction for each cell.
    # For each cell orientation o and each rel_dir r, abs_dir = rel_to_abs[o, r]
    # orientation: (n_plants, l_world, l_world)  -> expand to (n_plants,l,l,4)
    orient_exp = orientation.unsqueeze(-1).expand(-1, -1, -1, 4)  # (P,L,L,4)
    rel_dirs   = torch.arange(4, device=device).view(1, 1, 1, 4).expand_as(orient_exp)
    abs_dirs   = rel_to_abs[orient_exp, rel_dirs]  # (n_plants, l_world, l_world, 4)

    # Destination grid positions
    rows = torch.arange(l_world, device=device).view(1, l_world, 1, 1).expand(n_plants, l_world, l_world, 4)
    cols = torch.arange(l_world, device=device).view(1, 1, l_world, 1).expand(n_plants, l_world, l_world, 4)

    dest_i = rows + dir_delta[abs_dirs, 0]   # (P, L, L, 4)
    dest_j = cols + dir_delta[abs_dirs, 1]

    # Valid: source alive, signal above threshold, destination in bounds and empty
    in_bounds   = (dest_i >= 0) & (dest_i < l_world) & (dest_j >= 0) & (dest_j < l_world)
    above_thr   = div_signals > thr_division
    source_alive = alive.unsqueeze(-1).expand_as(div_signals)

    # Clamp for safe indexing (invalid entries will be masked out)
    dest_i_c = dest_i.clamp(0, l_world - 1)
    dest_j_c = dest_j.clamp(0, l_world - 1)

    p_idx = torch.arange(n_plants, device=device).view(n_plants, 1, 1, 1).expand_as(dest_i_c)
    dest_empty = ~alive[p_idx, dest_i_c, dest_j_c]   # (P, L, L, 4)

    valid = source_alive & above_thr & in_bounds & dest_empty  # (P, L, L, 4)

    # Per cell: keep only the strongest valid direction.
    # Set invalid entries to -inf so argmax ignores them.
    masked_signals = torch.where(valid, div_signals, torch.full_like(div_signals, float('-inf')))
    best_rel_dir   = masked_signals.argmax(dim=-1)   # (P, L, L)  relative dir index
    any_valid      = valid.any(dim=-1)               # (P, L, L)

    # Gather properties of the best direction for each cell
    P_idx  = torch.arange(n_plants, device=device).view(n_plants, 1, 1).expand(n_plants, l_world, l_world)
    I_idx  = torch.arange(l_world,  device=device).view(1, l_world, 1).expand(n_plants, l_world, l_world)
    J_idx  = torch.arange(l_world,  device=device).view(1, 1, l_world).expand(n_plants, l_world, l_world)

    best_abs_dir = abs_dirs[P_idx, I_idx, J_idx, best_rel_dir]     # (P, L, L)
    best_value   = div_signals[P_idx, I_idx, J_idx, best_rel_dir]  # (P, L, L)

    # Filter to cells with at least one valid direction
    mask = any_valid   # (P, L, L)

    return {
        'p':     P_idx[mask],
        'i':     I_idx[mask],
        'j':     J_idx[mask],
        'abs_d': best_abs_dir[mask],
        'value': best_value[mask],
    }


def resolve_race_conditions(
    candidates: dict,
    l_world: int,
    n_plants: int,
    device: torch.device,
) -> dict:
    """When two cells target the same square, keep the one with the higher signal.

    Uses a scatter-max approach: for each target square, retain only the
    candidate with the maximum value.

    Parameters
    ----------
    candidates : dict as returned by find_division_candidates
    l_world    : int
    n_plants   : int
    device     : torch.device

    Returns
    -------
    Filtered candidates dict (same structure).
    """
    if candidates['p'].numel() == 0:
        return candidates

    dir_delta = DIR_TO_DELTA.to(device)

    p     = candidates['p']
    i     = candidates['i']
    j     = candidates['j']
    abs_d = candidates['abs_d']
    value = candidates['value']

    # Destination squares
    dest_i = i + dir_delta[abs_d, 0]
    dest_j = j + dir_delta[abs_d, 1]

    # Encode each target as a flat index: plant * l_world^2 + i * l_world + j
    flat_target = p * (l_world * l_world) + dest_i * l_world + dest_j  # (N,)

    # For each unique flat_target keep the candidate with the max value.
    # Approach: sort by (flat_target, -value), then keep first occurrence of each target.
    order     = torch.argsort(flat_target * 1e12 - value.double())  # stable tie-break on value desc
    sorted_ft = flat_target[order]
    # Keep first occurrence of each target (= highest value after sorting)
    keep_mask = torch.ones(order.shape[0], dtype=torch.bool, device=device)
    keep_mask[1:] = sorted_ft[1:] != sorted_ft[:-1]

    keep_original = order[keep_mask]   # indices into original candidates

    return {k: v[keep_original] for k, v in candidates.items()}


def apply_divisions(
    state:      SimState,
    candidates: dict,
) -> SimState:
    """Apply all surviving division candidates as a batch update.

    For each division:
    - The mother cell's INTERNAL signals are halved.
    - The daughter cell is created with INTERNAL signals equal to the
      mother's post-halved values; all other output positions are zero.
    - The daughter's orientation is the argmax of the mother's
      DIVISION_DIRECTION directional outputs (0..3); tie / all-zero -> inherit mother.

    Parameters
    ----------
    state      : SimState  (modified in-place)
    candidates : dict as returned by resolve_race_conditions

    Returns
    -------
    state (same object, modified in-place)
    """
    if candidates['p'].numel() == 0:
        return state

    device    = state.alive.device
    l_world   = state.alive.shape[1]
    dir_delta = DIR_TO_DELTA.to(device)
    rel_to_abs = REL_TO_ABS.to(device)

    p     = candidates['p']
    i     = candidates['i']
    j     = candidates['j']
    abs_d = candidates['abs_d']

    # Destination squares
    dest_i = i + dir_delta[abs_d, 0]
    dest_j = j + dir_delta[abs_d, 1]

    # --- Mother: halve INTERNAL signals ---
    state.output[p, i, j, :, INTERNAL] /= 2.0

    # --- Daughter: set alive, copy halved INTERNAL signals, zero rest ---
    state.alive[p, dest_i, dest_j] = True
    # daughter output was already zero (never alive before); set INTERNAL
    state.output[p, dest_i, dest_j, :, INTERNAL] = state.output[p, i, j, :, INTERNAL]

    # --- Daughter orientation: argmax of mother's DIVISION_DIRECTION directional outputs ---
    # div_dir_signals: (N, 4)  -- the four relative-direction outputs of DIVISION_DIRECTION
    div_dir_signals = state.output[p, i, j, DIVISION_DIRECTION, :4]  # (N, 4)

    # Convert relative directions to absolute for the mother's orientation
    mother_orient = state.orientation[p, i, j]   # (N,)
    rel_dirs = torch.arange(4, device=device).unsqueeze(0).expand(p.shape[0], 4)  # (N,4)
    abs_dirs_mat = rel_to_abs[mother_orient.unsqueeze(-1).expand_as(rel_dirs), rel_dirs]  # (N,4)

    # argmax of div_dir_signals -> index into the 4 columns -> look up abs direction
    best_rel = div_dir_signals.argmax(dim=-1)   # (N,)  relative dir with highest value
    best_abs = abs_dirs_mat[torch.arange(p.shape[0], device=device), best_rel]  # (N,)

    # Tie-break: if all four values are zero, inherit mother's orientation
    all_zero = (div_dir_signals.sum(dim=-1) == 0.0)
    daughter_orient = torch.where(all_zero, mother_orient, best_abs)

    state.orientation[p, dest_i, dest_j] = daughter_orient

    return state

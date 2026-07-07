"""
Scoring via dual flood fill.

Score = number of live cells that are:
  (a) in the seed-connected component of live cells, AND
  (b) reachable from the world border through empty cells.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# 3x3 cross stencil (no corners) as a convolution kernel
_CROSS = torch.tensor(
    [[0., 1., 0.],
     [1., 1., 1.],
     [0., 1., 0.]],
).view(1, 1, 3, 3)


def _flood_fill(
    start_mask: Tensor,   # (n_plants, l_world, l_world) bool – seed pixels
    prop_mask:  Tensor,   # (n_plants, l_world, l_world) bool – pixels that can propagate
    max_iters:  int,
) -> Tensor:
    """Generic iterative flood fill via 2D convolution.

    Expands `start_mask` into connected regions of `prop_mask`.

    At each iteration:
        reachable = (conv(reachable) > 0) & prop_mask
    Repeats until stable or max_iters reached.

    Returns
    -------
    BoolTensor (n_plants, l_world, l_world): pixels reachable from start through prop.
    """
    device = start_mask.device
    cross = _CROSS.to(device)

    # Work in float for the convolution
    reachable = (start_mask & prop_mask).float()   # (P, L, L)

    # Add channel dim for F.conv2d: (P, 1, L, L)
    for _ in range(max_iters):
        prev = reachable
        expanded = F.conv2d(reachable.unsqueeze(1), cross, padding=1).squeeze(1)
        reachable = ((expanded > 0) & prop_mask).float()
        if torch.equal(reachable, prev):
            break

    return reachable.bool()


def flood_fill_empty(
    alive:   Tensor,   # (n_plants, l_world, l_world) bool
    l_world: int,
) -> Tensor:
    """Flood fill from world border through empty cells.

    Returns BoolTensor (n_plants, l_world, l_world):
    True where an empty cell is reachable from the border via empty cells.
    (Live cells on the exposed boundary are adjacent to such empty cells
    but are themselves not in this set — they are identified in score_plants.)
    """
    device = alive.device
    n_plants = alive.shape[0]

    empty = ~alive   # (P, L, L)

    # Border mask: all cells on the outermost ring
    border = torch.zeros(l_world, l_world, dtype=torch.bool, device=device)
    border[0, :]  = True
    border[-1, :] = True
    border[:, 0]  = True
    border[:, -1] = True
    border = border.unsqueeze(0).expand(n_plants, -1, -1)   # (P, L, L)

    max_iters = l_world * l_world
    return _flood_fill(border, empty, max_iters)


def flood_fill_live(
    alive:    Tensor,   # (n_plants, l_world, l_world) bool
    l_world:  int,
) -> Tensor:
    """Flood fill from seed position through live cells.

    Returns BoolTensor (n_plants, l_world, l_world):
    True for live cells in the connected component containing the seed.
    """
    device = alive.device
    n_plants = alive.shape[0]

    cx = l_world // 2
    cy = l_world // 2

    seed = torch.zeros(n_plants, l_world, l_world, dtype=torch.bool, device=device)
    seed[:, cx, cy] = True

    max_iters = l_world * l_world
    return _flood_fill(seed, alive, max_iters)


def score_plants(
    alive:   Tensor,   # (n_plants, l_world, l_world) bool
    l_world: int,
) -> Tensor:
    """Compute the score for each plant.

    Score = number of live cells that are both:
      - in the seed-connected live component
      - adjacent to empty cells reachable from the world border

    (A live cell is "adjacent to border-reachable empty space" if it neighbours
    a cell in the flood_fill_empty set.  We check this by dilating the empty
    fill by one step and intersecting with alive.)

    Returns
    -------
    LongTensor (n_plants,)
    """
    device = alive.device

    border_empty = flood_fill_empty(alive, l_world)   # (P, L, L) bool
    seed_live    = flood_fill_live(alive, l_world)    # (P, L, L) bool

    # A live cell scores if it is adjacent to border_empty (i.e., it is on the
    # exposed surface of the plant) AND is in the seed-connected component.
    # "Adjacent to border_empty" = dilate border_empty by one step and intersect with alive.
    cross  = _CROSS.to(device)
    dilated = F.conv2d(border_empty.float().unsqueeze(1), cross, padding=1).squeeze(1)
    on_surface = (dilated > 0) & alive   # (P, L, L)

    scoring_cells = on_surface & seed_live   # (P, L, L)

    return scoring_cells.long().sum(dim=(-2, -1))   # (P,)


def score_boundary_sides(
    alive:   Tensor,   # (n_plants, l_world, l_world) bool
    l_world: int,
) -> Tensor:
    """Score = number of exposed boundary *sides* (edges), not surface cells.

    For each live cell in the seed-connected component, count how many of its
    four neighbours are empty cells reachable from the world border, and sum
    over all such cells.  A live cell adjacent to two border-reachable empty
    cells contributes 2; two live cells adjacent to the same empty cell each
    contribute 1 (total 2).  This rewards total exposed perimeter length rather
    than the number of surface cells (cf. `score_plants`).

    Returns
    -------
    LongTensor (n_plants,)
    """
    device = alive.device

    border_empty = flood_fill_empty(alive, l_world)   # (P, L, L) bool
    seed_live    = flood_fill_live(alive, l_world)     # (P, L, L) bool

    # Count border-reachable empty 4-neighbours of every cell.  _CROSS includes
    # the centre, but a live cell is never itself in border_empty (it is not
    # empty), so the centre term contributes 0 at every scoring cell.
    cross = _CROSS.to(device)
    side_count = F.conv2d(border_empty.float().unsqueeze(1), cross, padding=1).squeeze(1)

    # Keep only sides belonging to seed-connected live cells, then sum.
    scoring_sides = side_count * seed_live.float()   # (P, L, L)

    return scoring_sides.sum(dim=(-2, -1)).long()   # (P,)

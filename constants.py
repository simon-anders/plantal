"""
Named constants for the artificial plant life simulation.
"""

import torch

# ---------------------------------------------------------------------------
# Signal indices
# ---------------------------------------------------------------------------
DIVISION_TRIGGER   = 0   # directional output in this signal triggers division
INITIAL_SIGNAL     = 1   # seeded to 1 in the initial seed cell at t=0
DIVISION_DIRECTION = 2   # determines daughter cell orientation on division

# ---------------------------------------------------------------------------
# Relative direction indices  (used for input/output tensor axis 4, 0..3)
# ---------------------------------------------------------------------------
APEX     = 0
BASE     = 1
LEFT     = 2
RIGHT    = 3
INTERNAL = 4

# ---------------------------------------------------------------------------
# Absolute direction indices
# ---------------------------------------------------------------------------
NORTH = 0
SOUTH = 1
WEST  = 2
EAST  = 3

# ---------------------------------------------------------------------------
# Rotation look-up tables
#
# REL_TO_ABS[orientation, rel_dir] -> abs_dir
#   Maps a relative direction (APEX/BASE/LEFT/RIGHT) to an absolute direction
#   given the cell's orientation.
#
#   orientation row order: NORTH, SOUTH, WEST, EAST
#   Columns: APEX, BASE, LEFT, RIGHT
#
#   North-pointing: APEX->N, BASE->S, LEFT->W, RIGHT->E  => [0,1,2,3]
#   South-pointing: APEX->S, BASE->N, LEFT->E, RIGHT->W  => [1,0,3,2]
#   West-pointing:  APEX->W, BASE->E, LEFT->S, RIGHT->N  => [2,3,1,0]
#   East-pointing:  APEX->E, BASE->W, LEFT->N, RIGHT->S  => [3,2,0,1]
# ---------------------------------------------------------------------------
REL_TO_ABS = torch.tensor([
    [0, 1, 2, 3],   # NORTH-pointing
    [1, 0, 3, 2],   # SOUTH-pointing
    [2, 3, 1, 0],   # WEST-pointing
    [3, 2, 0, 1],   # EAST-pointing
], dtype=torch.long)  # shape (4, 4)

# ABS_TO_REL[orientation, abs_dir] -> rel_dir
#   Inverse of REL_TO_ABS: maps absolute direction to relative direction.
#
#   North-pointing: N->APEX, S->BASE, W->LEFT,  E->RIGHT  => [0,1,2,3]
#   South-pointing: N->BASE, S->APEX, W->RIGHT, E->LEFT   => [1,0,3,2]
#   West-pointing:  N->RIGHT,S->LEFT, W->APEX,  E->BASE   => [3,2,0,1]
#   East-pointing:  N->LEFT, S->RIGHT,W->BASE,  E->APEX   => [2,3,1,0]
ABS_TO_REL = torch.tensor([
    [0, 1, 2, 3],   # NORTH-pointing
    [1, 0, 3, 2],   # SOUTH-pointing
    [3, 2, 0, 1],   # WEST-pointing
    [2, 3, 1, 0],   # EAST-pointing
], dtype=torch.long)  # shape (4, 4)

# NEGATE_DIR[abs_dir] -> opposite abs_dir
#   Swaps N<->S and W<->E so a signal "travelling north" arrives at
#   the south face of its destination.
NEGATE_DIR = torch.tensor([1, 0, 3, 2], dtype=torch.long)

# DIR_TO_DELTA[abs_dir] -> (di, dj) grid shift
#   NORTH: row-1, SOUTH: row+1, WEST: col-1, EAST: col+1
DIR_TO_DELTA = torch.tensor([
    [-1,  0],   # NORTH
    [ 1,  0],   # SOUTH
    [ 0, -1],   # WEST
    [ 0,  1],   # EAST
], dtype=torch.long)  # shape (4, 2)

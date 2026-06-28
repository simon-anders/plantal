# Artificial Plant Life

This is a plan for a simple software experiment for an artificial life (ALife) simulation.

Each plant will grow in its own "world"; so (for now) there is no competition between individuals.

---

## World & Grid

The world is a square grid of width and height `l_world`. Each grid square can contain a cell. At the beginning, all grid squares are empty ("dead") except for the middle one, which contains the initial seed cell, which is considered alive. Under certain conditions, a cell can divide and thus cause one of its neighbouring squares to become alive.

Time happens in discrete steps. In each step, each cell receives input from its neighbours and from its own previous state, and produces output to be stored as internal state and to be passed on to its neighbours in the next step. The output is calculated by passing the input through `n_layers` dense layers, all of the same size, which is also equal to the input and output size.

---

## Named Constants

The following named constants are used throughout. Using these names (rather than raw integers) is required in the implementation.

### Signal indices (`n_signal >= 3`)

| Name | Index | Role |
|---|---|---|
| `DIVISION_TRIGGER` | 0 | Output in this signal triggers cell division |
| `INITIAL_SIGNAL` | 1 | Seeded to 1 in the initial seed cell; no other special role |
| `DIVISION_DIRECTION` | 2 | Determines new daughter cell's orientation on division |

### Relative direction indices

| Name | Index | Meaning |
|---|---|---|
| `APEX` | 0 | The cell's apex (growth) direction |
| `BASE` | 1 | Opposite of apex |
| `LEFT` | 2 | Left when looking towards apex |
| `RIGHT` | 3 | Right when looking towards apex |
| `INTERNAL` | 4 | Internal state; does not propagate to neighbours |

### Absolute direction indices

| Name | Index |
|---|---|
| `NORTH` | 0 |
| `SOUTH` | 1 |
| `WEST` | 2 |
| `EAST` | 3 |

---

## Cell State & Polarisation

A cell's state is represented by a tensor of shape `(n_signal, 5)` that encapsulates both information exchange with neighbours and internal state.

Cells are polarised: each cell has one of the four absolute directions `{NORTH, SOUTH, WEST, EAST}` as its **apex** direction. The opposite direction is the **base**. The two remaining directions are **left** and **right** as seen when looking towards the apex. A cell is said to be, for example, "pointing west" if its apex direction is `WEST`.

There are `n_signal` signals. Each signal has 5 directions: `APEX`, `BASE`, `LEFT`, `RIGHT`, and `INTERNAL`. When describing input, these refer to the signal's value as stored in the previous step's output — either the `INTERNAL` output of the same cell, or the directed output arriving from the respective neighbour. The input is thus a tensor of shape `(n_signal, 5)`. The total input tensor across all cells has shape `(l_world, l_world, n_signal, 5)`. All dead cells have value zero.

---

## Network

There is a global network shared across all cells (but not across plants), with:

- **Weights tensor:** shape `(n_layers, n_signal, n_signal, 5, 5)`
- **Bias tensor:** shape `(n_layers, n_signal, 5)`

For each layer, the input tensor is contracted with the layer's weight tensor and the layer's bias tensor is added, following the PyTorch dense layer convention. This is passed through a **ReLU**, which is applied after **every** layer including the final one. Signals are therefore non-negative throughout.

To simulate `n_plants` worlds in parallel, the weight tensor is extended to shape `(n_plants, n_layers, n_signal, n_signal, 5, 5)` and the bias tensor to `(n_plants, n_layers, n_signal, 5)`.

---

## Step Execution Order

Each time step proceeds in the following order:

1. **Forward pass:** pass the input tensor through `n_layers` layers, applying ReLU after each layer (including the last), producing the output tensor.
2. **Budget clipping:** for each live cell, if the sum of all `n_signal × 5` output values exceeds `signal_max`, scale all values down proportionally so the sum equals `signal_max` exactly. (Set `signal_max = inf` to disable; this is the recommended initial setting.)
3. **Division processing** (see below).
4. **Signal routing:** transform the output tensor into the new input tensor by propagating directional signals to neighbouring cells (see below).
5. **Masking:** set all values at dead grid positions to zero.

---

## Signal Routing

### Relative → absolute direction permutations

The following permutation maps a cell's relative direction indices `(APEX, BASE, LEFT, RIGHT)` to absolute direction indices `(NORTH, SOUTH, WEST, EAST)`, depending on the cell's orientation:

| Cell orientation | APEX→ | BASE→ | LEFT→ | RIGHT→ | Permutation |
|---|---|---|---|---|---|
| North-pointing | N | S | W | E | 0, 1, 2, 3 |
| South-pointing | S | N | E | W | 1, 0, 3, 2 |
| West-pointing | W | E | S | N | 2, 3, 1, 0 |
| East-pointing | E | W | N | S | 3, 2, 0, 1 |

### Routing pipeline

For each live cell at position `(i, j)`:

1. Apply the sender's permutation to map each relative output direction (0–3) to an absolute direction.
2. Apply the **negation permutation** `(1, 0, 3, 2)` to get the face at which the signal arrives at the neighbouring cell (e.g. a signal travelling north arrives at the south face of the northern neighbour).
3. Apply the receiver's **inverse permutation** (see table below) to map the arriving absolute direction to the receiver's relative input index.
4. The `INTERNAL` component (index 4) stays in place and is not routed to any neighbour.

### Absolute → relative inverse permutations (for the receiving cell)

| Cell orientation | N→ | S→ | W→ | E→ | Inverse permutation |
|---|---|---|---|---|---|
| North-pointing | APEX | BASE | LEFT | RIGHT | 0, 1, 2, 3 |
| South-pointing | BASE | APEX | RIGHT | LEFT | 1, 0, 3, 2 |
| West-pointing | RIGHT | LEFT | APEX | BASE | 3, 2, 0, 1 |
| East-pointing | LEFT | RIGHT | BASE | APEX | 2, 3, 1, 0 |

Any signals routed to dead (empty) grid squares are discarded; those positions are set to zero in the masking step.

---

## Cell Division

Division is evaluated after budget clipping, on the output tensor.

### Identifying candidates

A cell at position `(i, j)` is a division candidate if any element of `output[i, j, DIVISION_TRIGGER, 0:4]` (the four directional components of the `DIVISION_TRIGGER` signal) exceeds the threshold `thr_division` and the target grid square in that absolute direction is empty.

### Per-cell rule

Only the **strongest** directional component of `output[:, :, DIVISION_TRIGGER, 0:4]` is considered per cell. If the corresponding target square is occupied, nothing happens for that cell.

### Race condition

If two cells both target the same empty square, the one with the **higher** signal value wins. All candidates are evaluated against the alive tensor **at the start of the step**, and all surviving divisions are applied as a **batch update**.

### Division mechanics

When a cell at position `(i, j)` divides in a given direction:

- **Mother cell:** `output[i, j, :, INTERNAL]` is halved (for all `n_signal` signals).
- **Daughter cell:** placed in the target square; its `output[:, :, :, INTERNAL]` is set to the mother's post-halved internal values; all other output positions are initialised to zero.
- **Budget is not re-checked** after division.

### Daughter cell orientation

The daughter's apex direction is the absolute direction corresponding to the **argmax** of `output[i, j, DIVISION_DIRECTION, 0:4]` (the four directional components of the mother's `DIVISION_DIRECTION` signal, excluding `INTERNAL`). In case of a tie, or if all four values are zero, the daughter **inherits the mother's orientation**.

Note: `DIVISION_DIRECTION` signals are also routed to neighbours as normal; the cells should learn to ignore this if needed.

---

## Initialization

To simulate `n_plants` worlds in parallel:

- **`alive`** tensor: shape `(n_plants, l_world, l_world)`, all zero except `alive[:, l_world//2, l_world//2] = 1`.
- **Input tensor:** shape `(n_plants, l_world, l_world, n_signal, 5)`, all zero except `input[:, l_world//2, l_world//2, INITIAL_SIGNAL, INTERNAL] = 1`.

The seed cell remains at position `(l_world//2, l_world//2)` throughout (cells do not move, only divide).

---

## World Size

The furthest a live cell can be from the seed after `n_steps` steps is `n_steps` (Manhattan distance). The plant therefore fits within a diamond of radius `n_steps`, which is contained in a square of side `2 * n_steps + 1`. The required minimum world size is:

```
l_world >= 2 * n_steps + 1
```

This guarantees that no plant ever touches the world border, so all border cells remain empty throughout. A runtime check that flags any border contact is recommended as a sanity guard.

---

## Scoring

After `n_steps` steps, each plant is scored as follows.

Two flood fills are performed:

1. **Border reachability:** starting from all border cells, propagate through **empty** cells only. This yields the set of live cells that are adjacent to (or reachable via) empty space connected to the border.
2. **Seed connectivity:** starting from the seed position `(l_world//2, l_world//2)`, propagate through **live** cells only. This yields the connected component of live cells containing the seed.

The plant's **score** is the number of live cells that appear in **both** sets.

This rewards cells that are part of the main plant body (connected to the seed) and that lie on the exposed boundary of the plant (reachable from the border via empty cells). Since there are no resource constraints, the selective pressure is entirely towards maximising boundary length, which is expected to favour fractal-like growth patterns.

**Implementation note:** Both flood fills can be implemented via iterated 2D convolution. For border reachability, convolve a "reachable" boolean tensor (initialised to 1 at the border) using the **empty** cell mask as the propagation mask, with a 3×3 cross stencil. For seed connectivity, convolve a "reachable" boolean tensor (initialised to 1 at the seed) using the **alive** tensor as the propagation mask.

---

## Evolution

We begin with a simple (1+1)-style generational mutation loop, to be extended later.

1. Initialise `n_plants` networks with weights drawn i.i.d. from a standard normal distribution.
2. Run `n_steps` steps and score all plants.
3. Keep the top `n_plants//2` plants by score, **exactly unchanged** (elitism).
4. Replace the bottom `n_plants//2` plants with mutated copies of the top plants: add i.i.d. normal noise with standard deviation `sd_mut` to all weight and bias values.
5. Repeat from step 2.

Possible future extensions:
- Crossover / mating between plant networks.
- CMA-ES or other gradient-free optimisers in place of the random walk.

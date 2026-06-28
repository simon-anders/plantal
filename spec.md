## Artificial plant life

This is a plan for a simple software experiment for an artificial life (AL) simulation.

Each plant will grow in its own "world"; so (for now) there is no competition between individuals.

The world is a square grid of width and height `l_world`. Each grid square can contain a cell. At the beginning, all grid squares are empty ("dead") except for the middle one, which contains the initial (seed) cell, which is considered alive. Under certain conditions, a cell can divide and thus cause one of its neighbor squares to become alive.

Time happens in discrete steps. In each step, each cell receives input from its neighbors and on its own previous state and produces output to be stored as internal state and to be passed on to its neighbors in the next step. The output is calculated by passing the input through `n_layers` dense layers, which all have the same size, which is also equal to input and output size.

A cell's state is represented by a tensor (shape described below) that encapsulates both the information exchange with its neighbors and the internal state.

The cells are polarized, i.e., they have a direction in the grid. One of the four directions `["north", "south", "east", "west"]` is called the cell the cell's "apex" direction, the opposite the "base" direction. The two remaining directions are labelled "left" and "right" as seen when looking towards the apex. A cell is said to be, e.g., "pointing west" if its apex direction is west.

There are `n_signal` state dimensions, each refered to as a "signal". The signal has 5 "directions", namely "apex", "base", "left", "right" and "internal", represented by the indices `0:5`. When describing input, these refer to the signal's values as it was stored in the previous step's output as the "internal" output or the directed output coming from the respective neighbors. Thus, the input is a tensor of shape `(n_signal,5)`. As we have one input for each cell, the total input tensor has shape `(l_world, l_world, n_signal, 5)`. For all dead cells, the value is zero.

There is a global network, with a weights tensor of shape `(n_layers, n_signal, n_signal, 5, 5)` and a bias tensor of shape `(n_layers, n_signal, 5)`. For each layer, we contract the input tensor with the layer's weight tensor and add the layer's bias tensor. This is pushed through a ReLU to become the step's output tensor. 

For the next step, the output tensor has to be turned into the new input tensor. This requires first masking out (setting to zero) all output values at dead grid squares. Then, the signals have to be moved to the respective neighbor. Consider, for example, a cell at grid position `(i,j)` that is pointing west. It's "apex" output signal, i.e., the subtensor `output[i,j,0,:]` will be moved to its western neighbor, at position `(i-1,j)`. If that cell is pointing north, it is receiving the signal from its right, i.e., we have `input[i-1,j,2,:] = output[i,j,0,:]`, where the indices `2` and `0` code for right and apex, respectively. The internal signals, at index 4, stay where they are. Note that this describes a reshuffling (permutation of tensor elements) with 1:1 correspondence of index tuples in output and input tensor. Any signals sent to empty squares have no receivers and therefore, as last step, the elements for empty grid positions are set to zero. to get the final input tensor for the next step.

Cell division: The signal with index 0 has a special meaning: If one of the ouput elements in `output[:,:,0,:4]` exceeds the threshold value `thr_division` and the grid square to which this signal would have been pushed is empty, the cell divides. This means that the grid square towards which the signal is directed becomes alive. The output value that caused the division is then set to zero. The output signals in the "internal" dimensions are divided by 2 and copied to the new cell's output tensor positions. 

Scoring: We simulate `n_steps` steps. Then, we determine for each alive cell whether it is connected to the world border, i.e., whether there is a continuous path from the cell to a border cell that only uses empty grid squares. The plant gets one point for each such cell. 

Note: To calculate this, we can do a flood fill via iterated 2D convolution of the alive tensor with a 3x3 cross stencil.

Initialization: To simulate `n_plants` worlds in parallel, we initialize `n_plants` worlds, i.e., a boolean `alive` tensor with size `(n_plants, l_world, l_world)` which is all zero except for `alive[:, l_world//2, l_world//2] = 1`, and an initial input tensor `(n_plants, l_world, l_world, n_signals, 5)` which is all zero except for `input[:, l_world//2, l_world//2, 1, 4] = 1` as initial internal state.

Evolution: As a first try, we do a simple biased random walk. We start with `n_plants` random networks, by initializing a tensor of size `(n_plants, n_layers, n_signal, n_signal, 5, 5)` with draws from a standard normal. We run `n_steps` steps and then score the plants. We keep the upper half, i.e., the `n_plants//2` plants with the highest scores and replace the others with copies of the kept plants, but modify the copies by adding iid normal random numbers with standard deviation `sd_mut`.

Later, we might try mating plants by crossing over their networks.

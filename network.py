"""
Network weights and forward pass.

Each plant has its own weight/bias tensors (no sharing across plants).
All cells within one plant share the same weights (standard cellular-automaton
neural-net convention).
"""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class Network:
    """Per-plant network parameters.

    Attributes
    ----------
    weights : FloatTensor (n_plants, n_layers, n_signal, n_signal, 5, 5)
        weights[p, l, s_out, s_in, d_out, d_in] is the weight connecting
        input signal s_in direction d_in to output signal s_out direction d_out
        in layer l of plant p.
    biases : FloatTensor (n_plants, n_layers, n_signal, 5)
        biases[p, l, s, d] is the bias for signal s direction d in layer l
        of plant p.
    """
    weights: Tensor   # (n_plants, n_layers, n_signal, n_signal, 5, 5)
    biases:  Tensor   # (n_plants, n_layers, n_signal, 5)


def init_network(
    n_plants: int,
    n_layers: int,
    n_signal: int,
    device:   torch.device = torch.device("cpu"),
) -> Network:
    """Initialise network weights i.i.d. from a standard normal distribution."""
    weights = torch.randn(n_plants, n_layers, n_signal, n_signal, 5, 5, device=device)
    biases  = torch.randn(n_plants, n_layers, n_signal, 5, device=device)
    return Network(weights=weights, biases=biases)


def forward(network: Network, x: Tensor) -> Tensor:
    """Run the forward pass for all plants and all cells simultaneously.

    Parameters
    ----------
    network : Network
    x : FloatTensor (n_plants, l_world, l_world, n_signal, 5)
        Input tensor (signals at each cell).

    Returns
    -------
    FloatTensor (n_plants, l_world, l_world, n_signal, 5)
        Output tensor after all layers with ReLU applied after each layer.

    Layer computation (following PyTorch Linear convention):
        out[s_out, d_out] = sum_{s_in, d_in} W[s_out, s_in, d_out, d_in]
                                             * in[s_in, d_in]
                          + bias[s_out, d_out]
    Using einsum:
        'pijab, plbacd -> pijcd'   (not quite — see below)

    We contract over (s_in=a, d_in=b) dimensions of x against the weight
    tensor W[p, l, s_out, s_in, d_out, d_in] = W[p,l,c,a,d,b].
    So the einsum is:  'pijab, plcadb -> pijcd'
    """
    n_plants, n_layers = network.weights.shape[:2]
    out = x  # (n_plants, l_world, l_world, n_signal, 5)

    for l in range(n_layers):
        W = network.weights[:, l]   # (n_plants, n_signal, n_signal, 5, 5)
        b = network.biases[:, l]    # (n_plants, n_signal, 5)

        # einsum: contract (s_in, d_in) axes of out against W
        # out  shape: (n_plants, l_world, l_world, s_in,  d_in )  -> p i j a b
        # W    shape: (n_plants, s_out, s_in, d_out, d_in)         -> p c a d b
        # result:     (n_plants, l_world, l_world, s_out, d_out)   -> p i j c d
        out = torch.einsum('pijab,pcadb->pijcd', out, W)

        # add bias: (n_plants, n_signal, 5) -> broadcast over (l_world, l_world)
        out = out + b[:, None, None, :, :]

        # ReLU after every layer (signals are non-negative throughout)
        out = torch.relu(out)

    return out

import torch
import torch.nn as nn

from src.utils import make_galaxy_labels_hierarchical


class ALReLU(nn.Module):
    """ Leaky Relu with absolute negative part
        https://arxiv.org/pdf/2012.07564v1.pdf
    """
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super(ALReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        x = torch.abs(torch.nn.functional.leaky_relu(x, self.negative_slope, self.inplace))
        return x


class Reshape(nn.Module):
    """ reshape tensor to given shape

        USAGE
        -----
        >>> torch.nn.Sequential(
        >>>    ...
        >>>    Reshape(*shape)
        >>>    ...
        >>> )
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class MaxOut(nn.Module):
    """ Maxout Layer: take max value from N_layers Linear layers
    take MaxPool1d of of single  Linear layer with N_layers*out_features nodes
    """
    def __init__(self, in_features: int, out_features: int, N_layers: int = 2, **kwargs):
        super(MaxOut, self).__init__()
        self.maxout = nn.Sequential(
            nn.Linear(in_features, N_layers*out_features, **kwargs),
            Reshape(1, N_layers*out_features),
            nn.MaxPool1d(N_layers),
            Reshape(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxout(x)
        return x


class GalaxyZooClassifier(nn.Module):

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self._model = nn.Sequential(
            MaxOut(n_in, n_out, bias=0.01),
            ALReLU(negative_slope=1e-2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(x)
        out = make_galaxy_labels_hierarchical(out)
        return out

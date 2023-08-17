from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Resnet(nn.Module):

    MODELS = {
        'resnet18': models.resnet18(pretrained=False),
        'resnet50': models.resnet50(pretrained=False)
    }

    def __init__(self,
                 base_model: str,
                 n_channels: int,
                 n_classes: int):
        super(Resnet, self).__init__()

        self._resnet = self.MODELS[base_model]
        num_ftrs = self._resnet.fc.in_features

        self._resnet.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                             stride=(2, 2), padding=(3, 3), bias=False)
        self._resnet.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._resnet(x)

    @property
    def fc(self):
        return self._resnet.fc

    @property
    def resnet(self):
        return self._resnet

    def children(self):
        return self._resnet.children()


class ResNetSimCLR(nn.Module):

    """ResNet based SimCLR model"""

    def __init__(self,
                 base_model: str,
                 n_channels: int,
                 out_dim: int):
        """
        Args:
            base_model: base model to use in SimCLR

            n_channels: number of channels in input image

            out_dim: size of output vector Z
        """

        super(ResNetSimCLR, self).__init__()

        resnet = Resnet(base_model, n_channels, 2)
        num_ftrs = resnet.fc.in_features

        self._features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self._l1 = nn.Linear(num_ftrs, num_ftrs)
        self._l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._features(x)
        h = h.squeeze()

        z = self._l1(h)
        z = F.relu(z)
        z = self._l2(z)
        return h, z

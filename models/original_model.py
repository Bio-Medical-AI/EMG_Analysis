import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import pad
from functools import partial
from typing import Any


class Conv2dCylindrical(nn.Module):
    """
    Conv2d Layer, with padding mode constant (value=0) for one axis and circular for another.

    Params:
        conv2d: Stored conv2d layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int or tuple[int, int],
                 stride: int or tuple[int, int] = 1,
                 padding: tuple[int, int] = 0,
                 dilation: int or tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device: Any = None,
                 dtype: Any = None) -> None:
        """

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to all four sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input channels to output channels
            bias: If True, adds a learnable bias to the output
            device: Device on which layer will be created
            dtype: Data type
        """
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                device=device,
                                dtype=dtype)
        self._horizontal_pad = partial(pad, pad=(0, 0, padding[1], padding[1]), mode='circular')
        self._vertical_pad = partial(pad, pad=(padding[0], padding[0]), mode='constant', value=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computing prediction for given matrix.
        Args:
            x: matrix to be convolved

        Returns:
            Tensor with performed convolution
        """

        return self.conv2d(self._vertical_pad(self._horizontal_pad(x)))


class OriginalModel(nn.Module):
    """
    Convolutional neural network presented in Article [1]

    Params:
        model: Sequence of layers that make up the model.
        num_classes: Number of classes in model output.
    """
    def __init__(self,
                 num_classes: int,
                 input_width: int,
                 input_height: int,
                 channels: int):
        """
        Args:
            num_classes: Number of classes predicted by network.
            input_width: Width of input picture
            input_height: Height of input picture
            channels: Number of channels in input picture
        """
        super().__init__()
        self.model = nn.Sequential(
            Conv2dCylindrical(channels, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Conv2dCylindrical(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(64 * input_width * input_height, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.weight_initialization()
        self.num_classes = num_classes

    def weight_initialization(self) -> None:
        """
        Initialization of model weights with the Xavier Uniform.
        """
        for layer in self.model:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computing prediction for given matrix.
        Args:
            x: Tensor representing picture

        Returns:
            Vector of values representing probability of picture being each class
        """
        return self.model(x)

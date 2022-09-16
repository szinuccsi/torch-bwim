import unittest

import torch
import torch.nn as nn

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.nets.modules.cnn.ConvBlock1D import ConvBlock1D


class MyTestCase(unittest.TestCase):

    batch_size = 32
    in_channels = 12
    out_channels = 18
    len = 255

    input_size = (batch_size, in_channels, len)

    input: torch.Tensor

    def setUp(self) -> None:
        self.input = torch.randn(*self.input_size)

    def test_same_mode_1(self):
        conv = ConvBlock1D(
            in_ch=self.in_channels, out_ch=self.out_channels,
            kernel_size=23, activation_function='relu'
        )
        output = conv.forward(self.input)
        TorchDataUtils.check_shape(output, expected_shape=(self.batch_size, self.out_channels, self.len))

    def test_same_mode_2(self):
        conv = ConvBlock1D(
            in_ch=self.in_channels, out_ch=self.out_channels,
            kernel_size=23, activation_function=nn.ReLU()
        )
        output = conv.forward(self.input)
        TorchDataUtils.check_shape(output, expected_shape=(self.batch_size, self.out_channels, self.len))

    def test_same_mode_dilation(self):
        conv = ConvBlock1D(
            in_ch=self.in_channels, out_ch=self.out_channels,
            kernel_size=23, activation_function=nn.ReLU(), dilation=2
        )
        output = conv.forward(self.input)
        TorchDataUtils.check_shape(output, expected_shape=(self.batch_size, self.out_channels, self.len))


if __name__ == '__main__':
    unittest.main()

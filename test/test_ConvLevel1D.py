import unittest

import torch

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.nets.modules.cnn.ConvLevel1D import ConvLevel1D


class ConvLevel1DTestCase(unittest.TestCase):

    batch_size = 32
    in_channels = 12
    out_channels = 18
    len = 255

    def setUp(self) -> None:
        self.cuda = False

    def net_create(self):
        return ConvLevel1D(
            config=ConvLevel1D.Config(
                in_ch=self.in_channels, out_ch=self.out_channels,
                kernel_size=31, layer_num=3,
                activation_function='relu', dropout_p=0.1
            )
        )

    def input_create(self): return torch.randn(self.batch_size, self.in_channels, self.len)

    def test_shape(self):
        net = self.net_create()
        input = self.input_create()
        output = net.forward(input)
        TorchDataUtils.check_shape(output, expected_shape=(self.batch_size, self.out_channels, self.len))


if __name__ == '__main__':
    unittest.main()

import unittest
import torch.nn as nn

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.modules.MultiLayerNet import MultiLayerNet


class MultiLayerNetTestCase(unittest.TestCase):

    BATCH_SIZE = 128
    INPUT_SIZE = 32
    HIDDEN_NEURONS = [212, 112]
    OUT_SIZE = 48

    DROPOUT_P = 0.1
    ACTIVATION_FUNCTION = 'relu'

    INPUT_SHAPE = (BATCH_SIZE, INPUT_SIZE)
    EXPECTED_SHAPE = (BATCH_SIZE, OUT_SIZE)

    def test_net_with_config_activation_function(self):
        net = MultiLayerNet(
            config=MultiLayerNet.Config(
                in_size=self.INPUT_SIZE, out_size=self.OUT_SIZE,
                hidden_neurons=self.HIDDEN_NEURONS, activation_function=self.ACTIVATION_FUNCTION,
                dropout_p=0.1, batch_norm=True
            )
        )
        in_tensor = torch.randn(self.INPUT_SHAPE)
        out_tensor = net.forward(in_tensor)
        TorchDataUtils.check_shape(out_tensor, expected_shape=self.EXPECTED_SHAPE)

    def test_net_with_constr_activation_function(self):
        net = MultiLayerNet(
            config=MultiLayerNet.Config(
                in_size=self.INPUT_SIZE, out_size=self.OUT_SIZE,
                hidden_neurons=self.HIDDEN_NEURONS,
            ),
            activation_function=nn.LeakyRelu()
        )
        in_tensor = torch.randn(self.INPUT_SHAPE)
        out_tensor = net.forward(in_tensor)
        TorchDataUtils.check_shape(out_tensor, expected_shape=self.EXPECTED_SHAPE)


if __name__ == '__main__':
    unittest.main()

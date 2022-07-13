import unittest

import numpy as np

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class NnModuleUtilsTestCase(unittest.TestCase):

    def test_padding_calculate(self):
        padding = NnModuleUtils.padding_calculate(kernel_size=3)
        self.assertEqual(padding, 1)

        padding = NnModuleUtils.padding_calculate(kernel_size=5, dilation=1)
        self.assertEqual(padding, 2)

        padding = NnModuleUtils.padding_calculate(kernel_size=5, dilation=2)
        self.assertEqual(padding, 4)

    ARR = [[1.0, 2.0, 3.0, 7.0], [2.0, 3.0, 5.0, 6.0]]

    ARR_SHAPE = (len(ARR), len(ARR[0]))

    def test_from_numpy(self):
        np_arr = np.asarray(self.ARR)

        t = NnModuleUtils.from_numpy(np_arr)
        TorchDataUtils.check_shape(t, expected_shape=np_arr.shape)

    def test_from_arr(self):
        t = NnModuleUtils.from_array(self.ARR)

        TorchDataUtils.check_shape(t, expected_shape=self.ARR_SHAPE)

    def test_from_nontensor_with_array(self):
        t = NnModuleUtils.from_nontensor(self.ARR)

        TorchDataUtils.check_shape(t, expected_shape=self.ARR_SHAPE)

    def test_from_nontensor_with_np_array(self):
        np_arr = np.asarray(self.ARR)
        t = NnModuleUtils.from_nontensor(np_arr)

        TorchDataUtils.check_shape(t, expected_shape=np_arr.shape)

    def test_from_nontensor_with_float(self):
        t = NnModuleUtils.from_nontensor(3.0)

        TorchDataUtils.check_shape(t, expected_shape=())

    def test_from_nontensor_with_int(self):
        t = NnModuleUtils.from_nontensor(3)

        TorchDataUtils.check_shape(t, expected_shape=())

    IN_SIZE = 3
    HIDDEN_NEURONS = [16, 32]
    OUT_SIZE = 7

    def test_get_neuron_counts(self):
        neuron_counts = NnModuleUtils.get_neuron_counts(
            in_size=self.IN_SIZE, hidden_neurons=self.HIDDEN_NEURONS,
            out_size=self.OUT_SIZE
        )
        self.assertEqual(len(neuron_counts), len(self.HIDDEN_NEURONS) + 2)
        self.assertEqual(neuron_counts[0], self.IN_SIZE)
        self.assertEqual(neuron_counts[-1], self.OUT_SIZE)
        for i in range(len(self.HIDDEN_NEURONS)):
            self.assertEqual(neuron_counts[i+1], self.HIDDEN_NEURONS[i])


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import torch

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

    def test_unsqueeze_tensors(self):
        t = torch.randn((128, 48, 32))
        u = torch.randn((16, 8))

        [res_0] = NnModuleUtils.unsqueeze_tensors([t], dim=0)
        self.assertTrue(TorchDataUtils.check_shape(res_0, expected_shape=(1, 128, 48, 32)))

        [res_10, res_11] = NnModuleUtils.unsqueeze_tensors([t, u], dim=1)
        self.assertTrue(TorchDataUtils.check_shape(res_10, expected_shape=(128, 1, 48, 32)))
        self.assertTrue(TorchDataUtils.check_shape(res_11, expected_shape=(16, 1, 8)))


if __name__ == '__main__':
    unittest.main()

import datetime
import unittest

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.interpolators.Interpolator1D import Interpolator1D
from torch_bwim.interpolators.Interpolator1DFunction import Interpolator1DFunction
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class Interpolator1DTestCase(unittest.TestCase):

    FIRST_SECTION = (-3.5, 1.5)
    FIRST_NUM = 100
    SECOND_SECTION = (2.0, 10.0)
    SECOND_NUM = 81

    TEST_SECTION = (-4.0, 12.0)
    TEST_NUM = 128

    xp: np.ndarray
    fp: np.ndarray

    epsilon = 1e-2

    def setUp(self):
        self.xp, self.fp = self.xp_fp_create()
        self.torch_interpolator = Interpolator1D.from_numpy(xp=self.xp, fp=self.fp)

    def xp_fp_create(self):
        first_section = np.linspace(start=self.FIRST_SECTION[0], stop=self.FIRST_SECTION[1], num=self.FIRST_NUM)
        second_section = np.linspace(start=self.SECOND_SECTION[0], stop=self.SECOND_SECTION[1], num=self.SECOND_NUM,
                                     endpoint=True)
        xp = np.concatenate((first_section, second_section))
        fp = np.concatenate((
            50.0 * np.sin(first_section * 10.0),
            np.square(second_section)
        ))
        self.grad_fp = np.gradient(fp, xp)
        return xp, fp

    def test_forward_one_seq(self):
        x = self.benchmark_section_create()
        np_f = np.interp(x, xp=self.xp, fp=self.fp)
        torch_f = self.torch_interpolator.forward(NnModuleUtils.from_array(x))
        TorchDataUtils.check_shape(torch_f, expected_shape=x.shape)
        plt.plot(x, np_f, label='numpy')
        plt.plot(x, torch_f.detach().numpy(), label='torch')
        plt.legend()
        plt.show()

        exp_f = np_f.tolist()
        act_f = torch_f.detach().numpy().tolist()
        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)

    def benchmark_section_create(self):
        return np.linspace(start=self.TEST_SECTION[0], stop=self.TEST_SECTION[1], num=self.TEST_NUM)

    def test_backward_one_seq(self):
        np_x = self.benchmark_section_create()
        np_grad_f = np.interp(np_x, xp=self.xp, fp=self.grad_fp)

        torch_x = NnModuleUtils.from_array(np_x)
        torch_x.requires_grad = True
        torch_f = self.torch_interpolator.forward(torch_x)
        loss = torch.sum(torch_f, dim=0)
        loss.backward()

        torch_grad_f = torch_x.grad

        plt.plot(np_x, np_grad_f, label='numpy')
        plt.plot(np_x, torch_grad_f.numpy(), label='torch')
        plt.legend()
        plt.show()

        exp_f = np_grad_f.tolist()
        act_f = torch_grad_f.detach().numpy().tolist()
        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)


if __name__ == '__main__':
    unittest.main()

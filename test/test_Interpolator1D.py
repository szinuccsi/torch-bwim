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
    cuda: bool

    def setUp(self):
        self.xp, self.fp = self.xp_fp_create()
        self.torch_interpolator = Interpolator1D.from_numpy(xp=self.xp, fp=self.fp)
        self.cuda = False

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

    def test_forward_one_function(self):
        x = self.benchmark_section_create()
        np_f = np.interp(x, xp=self.xp, fp=self.fp)
        torch_f = self.torch_interpolator.forward(NnModuleUtils.from_array(x, cuda=self.cuda))\
            .detach().cpu()

        TorchDataUtils.check_shape(torch_f, expected_shape=(x.shape[0], 1))

        exp_f = np_f.flatten()
        act_f = torch_f.detach().numpy().flatten()

        plt.plot(x, np_f, label='numpy')
        plt.plot(x, torch_f.numpy(), label='torch')
        plt.legend()
        plt.show()

        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)

    def benchmark_section_create(self):
        return np.linspace(start=self.TEST_SECTION[0], stop=self.TEST_SECTION[1], num=self.TEST_NUM)

    def test_backward_one_function(self):
        np_x = self.benchmark_section_create()
        np_grad_f = np.interp(np_x, xp=self.xp, fp=self.grad_fp)

        torch_x = NnModuleUtils.from_array(np_x, cuda=self.cuda)
        torch_x.requires_grad = True
        torch_f = self.torch_interpolator.forward(torch_x)
        loss = torch.sum(torch.sum(torch_f, dim=1), dim=0)
        loss.backward()

        torch_grad_f = torch_x.grad.detach().cpu()

        plt.plot(np_x, np_grad_f, label='numpy')
        plt.plot(np_x, torch_grad_f.numpy(), label='torch')
        plt.legend()
        plt.show()

        exp_f = np_grad_f.tolist()
        act_f = torch_grad_f.numpy()
        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)

    more_function_multiplier = 3.0

    def test_forward_more_function(self):
        self.torch_interpolator.append(
            fp=NnModuleUtils.from_array(self.more_function_multiplier * self.fp,
                                        cuda=self.cuda)
        )
        x = self.benchmark_section_create()
        np_f = np.interp(x, xp=self.xp, fp=self.fp)
        torch_f = self.torch_interpolator.forward(NnModuleUtils.from_array(x, cuda=self.cuda)) \
            .detach().cpu()

        TorchDataUtils.check_shape(torch_f, expected_shape=(x.shape[0], 2))

        exp_f = np.asarray([np_f, self.more_function_multiplier * np_f]).flatten()
        act_f = torch.transpose(torch_f, 0, 1).numpy().flatten()

        plt.plot(exp_f, label='numpy')
        plt.plot(act_f, label='torch')
        plt.legend()
        plt.show()

        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)

    def test_backward_more_function(self):
        self.torch_interpolator.append(
            fp=NnModuleUtils.from_array(self.more_function_multiplier * self.fp,
                                        cuda=self.cuda)
        )
        np_x = self.benchmark_section_create()
        np_grad_f = np.interp(np_x, xp=self.xp,
                              fp=(1. + self.more_function_multiplier) * self.grad_fp)

        torch_x = NnModuleUtils.from_array(np_x, cuda=self.cuda)
        torch_x.requires_grad = True
        torch_f = self.torch_interpolator.forward(torch_x)
        loss = torch.sum(torch.sum(torch_f, dim=1), dim=0)
        loss.backward()

        torch_grad_f = torch_x.grad.detach().cpu()

        plt.plot(np_x, np_grad_f, label='numpy')
        plt.plot(np_x, torch_grad_f.numpy(), label='torch')
        plt.legend()
        plt.show()

        exp_f = np_grad_f.tolist()
        act_f = torch_grad_f.numpy()
        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)

    A_weight = 0.33
    B_weight = 1.25

    def test_backward_more_function_weighted(self):
        self.torch_interpolator.append(
            fp=NnModuleUtils.from_array(self.more_function_multiplier * self.fp,
                                        cuda=self.cuda)
        )
        np_x = self.benchmark_section_create()
        np_grad_f = np.interp(np_x, xp=self.xp, fp=self.grad_fp)

        torch_x = NnModuleUtils.from_array(np_x, cuda=self.cuda)
        torch_x.requires_grad = True
        torch_f = self.torch_interpolator.forward(torch_x)
        torch_f_0 = torch.select(torch_f, dim=1, index=0)
        torch_f_1 = torch.select(torch_f, dim=1, index=1)
        loss = self.A_weight * torch.sum(torch_f_0) + self.B_weight * torch.sum(torch_f_1)
        loss.backward()

        np_grad_f = self.A_weight * np_grad_f + \
                    self.B_weight * self.more_function_multiplier * np_grad_f
        torch_grad_f = torch_x.grad.detach().cpu()

        plt.plot(np_x, np_grad_f, label='numpy')
        plt.plot(np_x, torch_grad_f.numpy(), label='torch')
        plt.legend()
        plt.show()

        exp_f = np_grad_f.tolist()
        act_f = torch_grad_f.numpy()
        for i in range(len(act_f)):
            self.assertAlmostEqual(act_f[i], exp_f[i], delta=self.epsilon)


if __name__ == '__main__':
    unittest.main()

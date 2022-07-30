import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as interpolate

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.helpers.RandomHelper import RandomHelper
from torch_bwim.interpolators.RectiBilinearInterpolate import RectiBilinearInterpolate
from torch_bwim.nets.NnModuleUtils import NnModuleUtils


class RectiBilinearInterpolateTestCase(unittest.TestCase):

    def grid_create(self, num_of_xs, num_of_ys):
        np.random.seed(42)
        distinct_xp = np.linspace(-2.0, 2.0, num=num_of_xs) ** 5
        distinct_yp = np.linspace(-3.0, 3.0, num=num_of_ys) ** 3
        mesh_x, mesh_y = np.meshgrid(distinct_xp, distinct_yp)
        mesh_z = np.sin(mesh_x / 10.0) + np.sin(mesh_y / 10.0) + np.cos((mesh_x + mesh_y) / 10.0)
        return distinct_xp, distinct_yp, mesh_x, mesh_y, mesh_z

    mesh_x: np.ndarray
    mesh_y: np.ndarray
    fp: np.ndarray
    distinct_xp: np.ndarray
    distinct_yp: np.ndarray
    epsilon = 1e-3

    cuda: bool

    def setUp(self) -> None:
        RandomHelper.set_random_state(42)
        self.distinct_xp, self.distinct_yp, self.mesh_x, self.mesh_y, self.fp = \
            self.grid_create(num_of_xs=5, num_of_ys=7)
        self.torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp),
            distinct_xp=NnModuleUtils.from_array(self.distinct_xp),
            distinct_yp=NnModuleUtils.from_array(self.distinct_yp)
        )
        self.cuda = False

    def test_control_points(self):
        x, y = self.mesh_x.flatten(), self.mesh_y.flatten()
        torch_f = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x, cuda=self.cuda),
            y=NnModuleUtils.from_array(y, cuda=self.cuda)
        ).detach().cpu()
        exp_shape = [s for s in x.shape]
        exp_shape.append(1)
        exp_shape = tuple(exp_shape)
        TorchDataUtils.check_shape(torch_f, expected_shape=exp_shape)

        exp_f = self.fp.flatten()
        act_f = torch_f.numpy().flatten()
        diff = np.abs(act_f - exp_f)
        for d in diff:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)
        self.plot(self.mesh_x, self.mesh_y, self.fp)
        self.plot(self.mesh_x, self.mesh_y, act_f.reshape(self.fp.shape))

    def plot(self, mesh_x, mesh_y, mesh_z):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(mesh_x, mesh_y, mesh_z)
        plt.show()

    def test_values_with_random_points_nearest(self):
        x_new = np.random.randn(32) * 7
        y_new = np.random.randn(32) * 7

        exp_f = interpolate.griddata(
            points=(self.mesh_x.flatten(), self.mesh_y.flatten()),
            values=self.fp.flatten(),
            xi=(x_new, y_new),
            fill_value=0.,
            method='nearest'
        )
        self.torch_interpolator.method = 'nearest'
        torch_f = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x_new, cuda=self.cuda),
            y=NnModuleUtils.from_array(y_new, cuda=self.cuda)
        ).detach().cpu()

        exp_f = exp_f.flatten()
        act_f = torch_f.numpy().flatten()
        diff = np.abs(act_f.flatten() - exp_f.flatten())
        for d in diff:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)

    SMALL_GRID_DISTINCT_XP = np.asarray([-1.0, 2.0, 3.0])
    SMALL_GRID_DISTINCT_YP = np.asarray([-2.0, +3.0])
    SMALL_GRID_FP = np.asarray([
        [2.0, 4.0, 3.0],
        [1.0, -1.0, 7.0]
    ])

    SMALL_GRID_X = np.asarray([
        -2.0, -1.5, -1.1, +1.0,
        +2.2, +0.5, +2.0, 3.2,
        +3.5, +3.2, 3.0,
        -0.5, +2.0, 2.75,
        -1.0, 0., 0.5, 2.25, 3.0, 2.75
    ])
    SMALL_GRID_Y = np.asarray([
        -3.0, +1.0, +3.1, +3.5,
        +3.2, -2.7, -2.1, -2.0,
        +0.1, +3.2, 0.5,
        -1.5, 0.5, 2.50,
        0.0, 3., -2.0, -2.0, 0.5, 3.0
    ])
    SMALL_GRID_EXP_F_FILL_MODE = np.asarray([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 5.0,
        +2.166, +1.5, +4.825,
        1.6, 1./3., 3.0, 3.75, 5.0, 5.0
    ])
    SMALL_GRID_EXP_F_EDGE_MODE = np.asarray([
        2.0, 1.4, 1.0, -0.33333,
        0.6, 3.0, 4.0, 3.0,
        4.68, 7.0, 5.0,
        +2.166, +1.5, +4.825,
        1.6, 1./3., 3.0, 3.75, 5.0, 5.0
    ])

    def test_linear_interpolation_on_small_grid_fill_zero(self):
        distinct_xp = self.SMALL_GRID_DISTINCT_XP
        distinct_yp = self.SMALL_GRID_DISTINCT_YP
        self.fp = self.SMALL_GRID_FP
        torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp, cuda=self.cuda),
            distinct_xp=NnModuleUtils.from_array(distinct_xp, cuda=self.cuda),
            distinct_yp=NnModuleUtils.from_array(distinct_yp, cuda=self.cuda)
        )
        self.torch_interpolator = torch_interpolator.cuda() if self.cuda else torch_interpolator
        self.small_grid_interpolation_test(exp_f=self.SMALL_GRID_EXP_F_FILL_MODE)

    def test_linear_interpolation_on_small_grid_edge(self):
        distinct_xp = self.SMALL_GRID_DISTINCT_XP
        distinct_yp = self.SMALL_GRID_DISTINCT_YP
        self.fp = self.SMALL_GRID_FP
        torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp, cuda=self.cuda),
            distinct_xp=NnModuleUtils.from_array(distinct_xp, cuda=self.cuda),
            distinct_yp=NnModuleUtils.from_array(distinct_yp, cuda=self.cuda),
            fill_mode='edge'
        )
        self.torch_interpolator = torch_interpolator.cuda() if self.cuda else torch_interpolator
        self.small_grid_interpolation_test(exp_f=self.SMALL_GRID_EXP_F_EDGE_MODE)

    def small_grid_interpolation_test(self, exp_f):
        x = self.SMALL_GRID_X
        y = self.SMALL_GRID_Y

        torch_f = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x, cuda=self.cuda),
            y=NnModuleUtils.from_array(y, cuda=self.cuda)
        ).detach().cpu()
        exp_f = exp_f.flatten()
        act_f = torch_f.detach().flatten()
        diff = np.abs(act_f.flatten() - exp_f.flatten())
        print(act_f.flatten())
        print(exp_f.flatten())
        for d in diff:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)

    more_grid_multiplier = 3.0

    def test_control_points_with_more_grid(self):
        self.torch_interpolator.append(
            fp=NnModuleUtils.from_array(self.more_grid_multiplier * self.fp,
                                        cuda=self.cuda)
        )
        x, y = self.mesh_x.flatten(), self.mesh_y.flatten()
        torch_f = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x, cuda=self.cuda),
            y=NnModuleUtils.from_array(y, cuda=self.cuda)
        ).detach().cpu()

        TorchDataUtils.check_shape(torch_f, expected_shape=(x.shape[0], 2))
        act_f = torch.transpose(torch_f, 0, 1).numpy().flatten()
        exp_f = np.asarray([self.fp, self.more_grid_multiplier * self.fp]).flatten()
        diff = np.abs(act_f - exp_f)
        for d in diff:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)

    grad_x_multiplier = 2.0
    grad_y_multiplier = 3.0

    def test_gradient_control_points(self):
        grad_x_fp = self.grad_x_multiplier * self.fp
        grad_y_fp = self.grad_y_multiplier * self.fp
        self.torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp, cuda=self.cuda),
            distinct_xp=NnModuleUtils.from_array(self.distinct_xp, cuda=self.cuda),
            distinct_yp=NnModuleUtils.from_array(self.distinct_yp, cuda=self.cuda),
            grad_x_fp=NnModuleUtils.from_array(grad_x_fp, cuda=self.cuda),
            grad_y_fp=NnModuleUtils.from_array(grad_y_fp, cuda=self.cuda)
        )

        x, y = self.mesh_x.flatten(), self.mesh_y.flatten()
        torch_x = NnModuleUtils.from_array(x, cuda=self.cuda)
        torch_y = NnModuleUtils.from_array(y, cuda=self.cuda)
        torch_x.requires_grad, torch_y.requires_grad = True, True
        torch_f = self.torch_interpolator.forward(x=torch_x, y=torch_y)
        loss = torch.sum(torch.sum(torch_f))
        loss.backward()

        TorchDataUtils.check_shape(torch_x.grad, expected_shape=x.shape)
        TorchDataUtils.check_shape(torch_y.grad, expected_shape=y.shape)

        act_grad_x = torch_x.grad.cpu().numpy().flatten()
        act_grad_y = torch_y.grad.cpu().numpy().flatten()
        exp_grad_x, exp_grad_y = grad_x_fp.flatten(), grad_y_fp.flatten()
        diff_grad_x = np.abs(act_grad_x - exp_grad_x)
        diff_grad_y = np.abs(act_grad_y - exp_grad_y)
        for d in diff_grad_x:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)
        for d in diff_grad_y:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)

    def test_gradient_on_small_grid(self):
        pass

    def test_gradient_more_grids(self):
        pass


if __name__ == '__main__':
    unittest.main()

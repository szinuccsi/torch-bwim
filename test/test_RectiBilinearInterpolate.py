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

    def setUp(self) -> None:
        RandomHelper.set_random_state(42)
        self.distinct_xp, self.distinct_yp, self.mesh_x, self.mesh_y, self.fp = \
            self.grid_create(num_of_xs=5, num_of_ys=7)
        self.torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp),
            distinct_xp=NnModuleUtils.from_array(self.distinct_xp),
            distinct_yp=NnModuleUtils.from_array(self.distinct_yp)
        )

    def test_control_points(self):
        x, y = self.mesh_x.flatten(), self.mesh_y.flatten()
        res = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x),
            y=NnModuleUtils.from_array(y)
        ).detach().numpy()
        exp_shape = [s for s in x.shape]
        exp_shape.append(1)
        exp_shape = tuple(exp_shape)
        TorchDataUtils.check_shape(res, expected_shape=exp_shape)
        diff = np.abs(res.flatten() - self.fp.flatten())
        for d in diff:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)

        self.plot(self.mesh_x, self.mesh_y, self.fp)
        self.plot(self.mesh_x, self.mesh_y, res.reshape(self.fp.shape))

    def plot(self, mesh_x, mesh_y, mesh_z):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(mesh_x, mesh_y, mesh_z)
        plt.show()

    def test_values_with_random_points(self):
        x_new = np.random.randn(32) * 7
        y_new = np.random.randn(32) * 7

        exp = interpolate.griddata(
            points=(self.mesh_x.flatten(), self.mesh_y.flatten()),
            values=self.fp.flatten(),
            xi=(x_new, y_new),
            fill_value=0.,
            method='nearest'
        )
        self.torch_interpolator.method = 'nearest'
        res = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x_new),
            y=NnModuleUtils.from_array(y_new)
        ).detach().numpy().flatten()
        diff = np.abs(res.flatten() - exp.flatten())
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
        +3.5, +3.2,
        -0.5, +2.0, 2.75
    ])
    SMALL_GRID_Y = np.asarray([
        -3.0, +1.0, +3.1, +3.5,
        +3.2, -2.7, -2.1, -2.0,
        +0.1, +3.2,
        -1.5, 0.5, 2.50]
    )
    SMALL_GRID_EXP = np.asarray([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        +2.166, +1.5, +4.825
    ])

    def test_linear_interpolation_on_small_grid(self):
        distinct_xp = self.SMALL_GRID_DISTINCT_XP
        distinct_yp = self.SMALL_GRID_DISTINCT_YP
        self.fp = self.SMALL_GRID_FP
        self.torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp),
            distinct_xp=NnModuleUtils.from_array(distinct_xp),
            distinct_yp=NnModuleUtils.from_array(distinct_yp)
        )

        x = self.SMALL_GRID_X
        y = self.SMALL_GRID_Y
        exp = self.SMALL_GRID_EXP

        res = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x),
            y=NnModuleUtils.from_array(y)
        ).detach().numpy()
        diff = np.abs(res.flatten() - exp.flatten())
        print(res.flatten())
        print(exp.flatten())
        for d in diff:
            self.assertAlmostEqual(d, 0., delta=self.epsilon)

    def test_control_points_with_more_grid(self):
        self.torch_interpolator.add_numpy(fp=2 * self.fp)
        x, y = self.mesh_x.flatten(), self.mesh_y.flatten()
        res = self.torch_interpolator.forward(
            x=NnModuleUtils.from_array(x),
            y=NnModuleUtils.from_array(y)
        ).detach().numpy()

        exp_shape = list(x.shape)
        exp_shape.append(2)
        exp_shape = tuple(exp_shape)
        TorchDataUtils.check_shape(res, expected_shape=exp_shape)
        for i in range(2):
            diff = np.abs(res[:, i].flatten() - (i+1) * self.fp.flatten())
            for d in diff:
                self.assertAlmostEqual(d, 0., delta=self.epsilon)

    def test_gradient_control_points(self):
        grad_x_fp = 2.0 * self.fp
        grad_y_fp = 3.0 * self.fp
        self.torch_interpolator = RectiBilinearInterpolate(
            fp=NnModuleUtils.from_array(self.fp),
            distinct_xp=NnModuleUtils.from_array(self.distinct_xp),
            distinct_yp=NnModuleUtils.from_array(self.distinct_yp),
            grad_x_fp=NnModuleUtils.from_array(grad_x_fp),
            grad_y_fp=NnModuleUtils.from_array(grad_y_fp)
        )

        x, y = self.mesh_x.flatten(), self.mesh_y.flatten()
        torch_x, torch_y = NnModuleUtils.from_array(x), NnModuleUtils.from_array(y)
        torch_x.requires_grad, torch_y.requires_grad = True, True
        res = self.torch_interpolator.forward(x=torch_x, y=torch_y)
        loss = torch.sum(torch.sum(res))
        loss.backward()

        TorchDataUtils.check_shape(torch_x.grad, expected_shape=x.shape)
        TorchDataUtils.check_shape(torch_y.grad, expected_shape=y.shape)
        diff_grad_x = np.abs(torch_x.grad.detach().numpy().flatten() - grad_x_fp.flatten())
        diff_grad_y = np.abs(torch_y.grad.detach().numpy().flatten() - grad_y_fp.flatten())
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

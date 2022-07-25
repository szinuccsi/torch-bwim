import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch_bwim.interpolators.RectiBilinearInterpolate import RectiBilinearInterpolate


class RectiBilinearInterpolateTestCase(unittest.TestCase):
    def _value(self, mesh_x, mesh_y):
        return np.sin(mesh_x / 10.0) + np.sin(mesh_y / 10.0) + np.cos((mesh_x + mesh_y) / 10.0)

    def _grad(self, mesh_x, mesh_y):
        return np.cos(mesh_x / 10.0) * 1 / 10 - np.sin((mesh_x + mesh_y) / 10.0) / 10.0, \
               np.cos(mesh_y / 10.0) * 1 / 10.0 - np.sin((mesh_x + mesh_y) / 10.0) / 10.0

    def _create_surface(self, num_of_xs, num_of_ys):
        np.random.seed(42)
        distinct_xs = np.linspace(-2.0, 2.0, num=num_of_xs) ** 5
        distinct_ys = np.linspace(-3.0, 3.0, num=num_of_ys) ** 3
        mesh_x, mesh_y = np.meshgrid(distinct_xs, distinct_ys)
        mesh_z = self._value(mesh_x, mesh_y)
        return distinct_xs, distinct_ys, mesh_x, mesh_y, mesh_z

    def test_control_points(self):
        distinct_xs, distinct_ys, ctrl_xs, ctrl_ys, ctrl_values = self._create_surface(num_of_xs=5, num_of_ys=7)
        my_torch_interpolator = RectiBilinearInterpolate(
            ctrl_values=self._to_tensor(np.asarray([ctrl_values])),
            distinct_xs=self._to_tensor(distinct_xs),
            distinct_ys=self._to_tensor(distinct_ys),
            ctrl_gradient_x=None, ctrl_gradient_y=None,
            method='linear'
        )
        res = my_torch_interpolator.forward(
            self._to_tensor(ctrl_xs.flatten()), self._to_tensor(ctrl_ys.flatten())
        ).detach().numpy()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(ctrl_xs, ctrl_ys, ctrl_values)
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        res = res.reshape(ctrl_values.shape)
        ax.plot_surface(ctrl_xs, ctrl_ys, res)
        plt.show()
        diff = np.abs(res - ctrl_values)
        self.assertTrue(np.all(diff < 1e-6 * np.ones_like(diff)))

    def _to_tensor(self, np_array):
        return torch.from_numpy(np_array.astype(np.float32))

    def test_values_with_random_points(self):
        distinct_xs, distinct_ys, ctrl_xs, ctrl_ys, ctrl_values = self._create_surface(num_of_xs=5, num_of_ys=7)
        x_new = np.random.randn(32) * 4
        y_new = np.random.randn(32) * 4

        z_new = interpolate.griddata(
            points=(ctrl_xs.flatten(), ctrl_ys.flatten()),
            values=ctrl_values.flatten(),
            xi=(x_new, y_new),
            fill_value=0.,
            method='nearest'
        )

        my_torch_interpolator = RectiBilinearInterpolate(
            ctrl_values=self._to_tensor(np.asarray([ctrl_values])),
            distinct_xs=self._to_tensor(distinct_xs),
            distinct_ys=self._to_tensor(distinct_ys),
            ctrl_gradient_x=None, ctrl_gradient_y=None,
            method='linear'
        )
        res = my_torch_interpolator.forward(
            self._to_tensor(x_new),
            self._to_tensor(y_new)
        )

        exp = z_new
        res = res.detach().numpy()[0]
        print(res)
        print(exp)
        diff = np.abs(res - exp)
        print(diff)
        print(np.max(diff))
        self.assertTrue(np.allclose(diff, np.zeros_like(diff), atol=2e-2))

    def test_gradient_with_control_points(self):
        distinct_xs, distinct_ys, ctrl_xs, ctrl_ys, ctrl_values = self._create_surface(num_of_xs=300, num_of_ys=400)

        np_grad_x, np_grad_y = self._grad(ctrl_xs, ctrl_ys)

        grad_numpy = np.gradient(
            ctrl_values,
            distinct_ys,
            distinct_xs
        )

        my_torch_interpolator = RectiBilinearInterpolate(
            ctrl_values=self._to_tensor(np.asarray([ctrl_values])),
            distinct_xs=self._to_tensor(distinct_xs),
            distinct_ys=self._to_tensor(distinct_ys),
            ctrl_gradient_x=self._to_tensor(np.asarray([grad_numpy[1]])),
            ctrl_gradient_y=self._to_tensor(np.asarray([grad_numpy[0]])),
            method='linear'
        )
        xs, ys = self._to_tensor(ctrl_xs.flatten()), self._to_tensor(ctrl_ys.flatten())
        xs.requires_grad, ys.requires_grad = True, True
        res = my_torch_interpolator.forward(
            xs, ys
        )
        sum_res = torch.sum(torch.flatten(res, start_dim=0), dim=0)
        sum_res.backward()
        gradient_x, gradient_y = xs.grad.numpy(), ys.grad.numpy()
        gradient_x = gradient_x.reshape(np_grad_x.shape)
        gradient_y = gradient_y.reshape(np_grad_y.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(f'Truth')
        ax.plot_surface(ctrl_xs, ctrl_ys, np_grad_x)
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(f'numpy.gradient')
        ax.plot_surface(ctrl_xs, ctrl_ys, grad_numpy[1])
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(f'RectiBilinearInterpolate')
        ax.plot_surface(ctrl_xs, ctrl_ys, gradient_x)
        plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(f'Truth')
        ax.plot_surface(ctrl_xs, ctrl_ys, np_grad_y)
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(f'numpy.gradient')
        ax.plot_surface(ctrl_xs, ctrl_ys, grad_numpy[0])
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(f'RectiBilinearInterpolate')
        ax.plot_surface(ctrl_xs, ctrl_ys, gradient_y)
        plt.show()

    def test_gradient_with_random_points(self):
        pass


if __name__ == '__main__':
    unittest.main()

import unittest

from test_RectiBilinearInterpolate import RectiBilinearInterpolateTestCase


class RectiBilinearInterpolateCudaTestCase(RectiBilinearInterpolateTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.cuda = True
        self.torch_interpolator = self.torch_interpolator.cuda()

    def test_control_points(self):
        super().test_control_points()

    def test_values_with_random_points_nearest(self):
        super().test_values_with_random_points_nearest()

    def test_linear_interpolation_on_small_grid_fill_zero(self):
        super().test_linear_interpolation_on_small_grid_fill_zero()

    def test_linear_interpolation_on_small_grid_edge(self):
        super().test_linear_interpolation_on_small_grid_edge()

    def test_control_points_with_more_grid(self):
        super().test_control_points_with_more_grid()

    def test_gradient_control_points(self):
        super().test_gradient_control_points()

    def test_merge(self):
        super().test_merge()


if __name__ == '__main__':
    unittest.main()

import unittest

from test_Interpolator1D import Interpolator1DTestCase


class Interpolator1DCudaTestCase(Interpolator1DTestCase):

    def setUp(self):
        super().setUp()
        self.cuda = True
        self.torch_interpolator = self.torch_interpolator.cuda()

    def test_forward_one_function(self):
        super().test_forward_one_function()

    def test_backward_one_function(self):
        super().test_backward_one_function()

    def test_forward_more_function(self):
        super().test_forward_more_function()

    def test_backward_more_function(self):
        super().test_backward_more_function()

    def test_backward_more_function_weighted(self):
        super().test_backward_more_function_weighted()


if __name__ == '__main__':
    unittest.main()

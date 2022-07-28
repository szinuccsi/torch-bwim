import unittest

from test_Interpolator1D import Interpolator1DTestCase


class Interpolator1DTestCudaCase(Interpolator1DTestCase):

    def setUp(self):
        super().setUp()
        self.cuda = True
        self.torch_interpolator = self.torch_interpolator.cuda()

    def test_forward_one_seq(self):
        super().test_forward_one_seq()

    def test_backward_one_seq(self):
        super().test_backward_one_seq()


if __name__ == '__main__':
    unittest.main()

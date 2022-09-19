import unittest

from test_ConvLevel1D import ConvLevel1DTestCase


class ConvLevel1DCudaTestCase(ConvLevel1DTestCase):

    def net_create(self):
        net = super().net_create()
        return net.cuda()

    def input_create(self):
        input = super().input_create()
        return input.cuda()

    def test_shape(self):
        super().test_shape()


if __name__ == '__main__':
    unittest.main()

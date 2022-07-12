import unittest
import torch.nn as nn

from torch_bwim.nets.ActivationFunctions import ActivationFunctions


class ActivationFunctionsTestCase(unittest.TestCase):

    def test_type(self):
        self.assertTrue(isinstance(ActivationFunctions.get_function(ActivationFunctions.Types.ReLU), nn.ReLU))
        self.assertTrue(isinstance(ActivationFunctions.get_function(ActivationFunctions.Types.LeakyReLU), nn.LeakyReLU))
        self.assertTrue(isinstance(ActivationFunctions.get_instance().
                                   create(ActivationFunctions.Types.Tanh), nn.Tanh))
        self.assertTrue(isinstance(ActivationFunctions.get_instance().
                                   create(ActivationFunctions.Types.Sigmoid), nn.Sigmoid))

    WORD1 = 'key'
    WORD2 = 'mock_object'

    def test_registration(self):
        ActivationFunctions.get_instance().register_activation_function(self.WORD1, function_factory=lambda: self.WORD2)

        self.assertEqual(ActivationFunctions.get_instance().create(self.WORD1), self.WORD2)


if __name__ == '__main__':
    unittest.main()

import torch.nn as nn


class ActivationFunctions(object):

    class Types(object):
        LeakyReLU = 'leaky_relu'
        ReLU = 'relu'
        Sigmoid = 'sigmoid'
        Tanh = 'tanh'
        Sinh = 'sinh'

    _instance = None

    @staticmethod
    def get_instance():
        if ActivationFunctions._instance is None:
            ActivationFunctions._instance = ActivationFunctions(
                activation_function_map={
                    ActivationFunctions.Types.ReLU: nn.ReLU,
                    ActivationFunctions.Types.LeakyReLU: nn.LeakyReLU,
                    ActivationFunctions.Types.Sigmoid: nn.Sigmoid,
                    ActivationFunctions.Types.Tanh: nn.Tanh,
                    ActivationFunctions.Types.Sinh: nn.Sinh
                }
            )
        return ActivationFunctions._instance

    def __init__(self, activation_function_map: dict):
        super().__init__()
        self.activation_function_map = activation_function_map

    def create(self, key: str) -> nn.Module:
        return self.activation_function_map[key]()

    def register_activation_function(self, key: str, function_factory):
        self.activation_function_map[key] = function_factory

from torch import optim

from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase


class SGDFactory(OptimizerFactoryBase):

    class Config(OptimizerFactoryBase.Config):
        def __init__(self, learning_rate, weight_decay, momentum, nesterov):
            super().__init__(optimizer_type=self.get_optimizer_type(), learning_rate=learning_rate)
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.nesterov = nesterov

        @classmethod
        def get_optimizer_type(cls) -> str:
            return 'SGD'

    def create_optimizer(self, parameters, config):
        return optim.SGD(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                         momentum=config.momentum,
                         nesterov=config.nesterov)

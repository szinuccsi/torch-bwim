from torch import optim

from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase


class AdamFactory(OptimizerFactoryBase):

    class Config(OptimizerFactoryBase.Config):
        def __init__(self, learning_rate, weight_decay, amsgrad=False):
            super().__init__(optimizer_type=self.get_optimizer_type(), learning_rate=learning_rate)
            self.weight_decay = weight_decay
            self.amsgrad = amsgrad

        @classmethod
        def get_optimizer_type(cls) -> str:
            return 'Adam'

    def create_optimizer(self, parameters, config):
        return optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                          amsgrad=config.amsgrad)

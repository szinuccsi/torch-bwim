class OptimizerFactoryBase(object):

    class Config(object):
        def __init__(self, optimizer_type: str, learning_rate):
            super().__init__()
            self.optimizer_type = optimizer_type
            self.learning_rate = learning_rate

        @classmethod
        def get_optimizer_type(cls) -> str:
            raise RuntimeError(f'Invalid optimizer_type (AbstractOptimizerFactory)')

    def create(self, parameters, config: Config):
        pass

from torch_bwim.helpers.PersistHelper import PersistHelper
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.optimizers.modules.AdamFactory import AdamFactory
from torch_bwim.optimizers.modules.SGDFactory import SGDFactory


class OptimizerBuilder(object):

    _instance = None

    @staticmethod
    def get_instance(force_new_instance=False):
        if (OptimizerBuilder._instance is None) or force_new_instance:
            OptimizerBuilder._instance = OptimizerBuilder()
        return OptimizerBuilder._instance

    def __init__(self):
        super().__init__()
        self.optimizer_factories = {}
        self.register_optimizer(SGDFactory.Config.get_optimizer_type(), SGDFactory())
        self.register_optimizer(AdamFactory.Config.get_optimizer_type(), AdamFactory())

    def register_optimizer(self, key: str, optimizer_factory: OptimizerFactoryBase):
        self.optimizer_factories[key] = optimizer_factory

    def create(self, parameters, config: OptimizerFactoryBase.Config):
        factory = self.optimizer_factories[config.optimizer_type]
        return factory.create_optimizer(parameters=parameters, config=config)

    @classmethod
    def create_optimizer(cls, parameters, config: OptimizerFactoryBase.Config):
        cls.get_instance().create(parameters, config=config)

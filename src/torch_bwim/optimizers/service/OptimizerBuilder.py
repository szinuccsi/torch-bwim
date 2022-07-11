from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.optimizers.modules.AdamFactory import AdamFactory
from torch_bwim.optimizers.modules.SGDFactory import SGDFactory


class OptimizerBuilder(object):

    _instance = None

    @classmethod
    def get_config_file(cls, filename=None):
        if filename is None:
            return "optimizer_config.json"
        return filename

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

    @staticmethod
    def create(parameters, config: OptimizerFactoryBase.Config):
        return OptimizerBuilder.get_instance().create(parameters, config)

    @classmethod
    def save_config(cls, optimizer_config: OptimizerFactoryBase.Config, save_folder_path, filename=None):
        return JsonHelper.save_json(
            data=optimizer_config,
            path=os.path.join(save_folder_path, cls.get_config_file(filename))
        )

    @classmethod
    def load_config(cls, save_folder_path, filename=None):
        return JsonHelper.load_json_to_object(path=os.path.join(save_folder_path, cls.get_config_file(filename)))

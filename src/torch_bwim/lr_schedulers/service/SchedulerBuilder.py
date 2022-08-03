from torch_bwim.lr_schedulers.modules.CosAnnealingScheduler import CosAnnealingScheduler
from torch_bwim.lr_schedulers.modules.NullScheduler import NullScheduler
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.lr_schedulers.SchedulerBase import SchedulerBase


class SchedulerBuilder(object):

    _instance = None

    @staticmethod
    def get_instance(new_instance=False):
        if (SchedulerBuilder._instance is None) or new_instance:
            SchedulerBuilder._instance = SchedulerBuilder()
        return SchedulerBuilder._instance

    def __init__(self):
        super().__init__()
        self.scheduler_class = {}
        self.register_scheduler(CosAnnealingScheduler)
        self.register_scheduler(NullScheduler)

    def register_scheduler(self, cls):
        self.scheduler_class[cls.Config.get_scheduler_type()] = cls

    def create(self, config: SchedulerBase.Config, optimizer, optimizer_config: OptimizerFactoryBase.Config):
        return self.scheduler_class[config.scheduler_type](
            config=config,
            optimizer=optimizer, optimizer_config=optimizer_config
        )

    @classmethod
    def create_scheduler(cls, config: SchedulerBase.Config, optimizer, optimizer_config: OptimizerFactoryBase.Config):
        return cls.get_instance().create(
            config=config, optimizer=optimizer,
            optimizer_config=optimizer_config
        )

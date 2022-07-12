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

    def register_scheduler(self, cls):
        self.scheduler_class[cls.Config.get_scheduler_type()] = cls

    def create(self, optimizer, optimizer_config: OptimizerFactoryBase.Config, scheduler_config: SchedulerBase.Config):
        return self.scheduler_class[scheduler_config.scheduler_type](
            optimizer=optimizer, optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )

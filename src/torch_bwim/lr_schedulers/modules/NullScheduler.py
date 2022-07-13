from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.lr_schedulers.SchedulerBase import SchedulerBase


class NullScheduler(SchedulerBase):

    class Config(SchedulerBase.Config):
        def __init__(self):
            super().__init__(scheduler_type=self.get_scheduler_type(), step_period=0)

        @classmethod
        def get_scheduler_type(cls):
            return 'NullScheduler'

    def __init__(self, optimizer, optimizer_config: OptimizerFactoryBase.Config, config: Config):
        super().__init__(optimizer=optimizer, optimizer_config=optimizer_config, config=config)

    def step(self, batch_size, t: torch.Tensor=None):
        pass

    def get_last_lr(self):
        return self.optimizer_config.learning_rate

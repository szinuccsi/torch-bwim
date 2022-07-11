from torch_bwim.lr_schedulers.SchedulerBase import SchedulerBase
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase


class CosAnnealingScheduler(SchedulerBase):

    class Config(SchedulerBase.Config):
        def __init__(self, step_period, annealing_period_in_steps, lr_ratio):
            super().__init__(scheduler_type=self.get_scheduler_type(), step_period=step_period)
            self.annealing_period_in_steps = annealing_period_in_steps
            self.lr_ratio = lr_ratio

        @classmethod
        def get_scheduler_type(cls):
            return 'CosAnnealingScheduler'

    def __init__(self, optimizer, optimizer_config: OptimizerFactoryBase.Config, scheduler_config: Config):
        super().__init__(optimizer=optimizer, optimizer_config=optimizer_config, scheduler_config=scheduler_config)
        self.scheduler_config = scheduler_config
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.scheduler_config.annealing_period_in_steps,
            eta_min=optimizer_config.learning_rate * self.scheduler_config.lr_ratio
        )

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

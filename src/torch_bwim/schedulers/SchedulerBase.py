from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase


class SchedulerBase(object):

    class Config(object):
        def __init__(self, scheduler_type: str, step_period):
            super().__init__()
            self.scheduler_type = scheduler_type
            self.step_period = step_period

        def get_scheduler_type(self):
            raise RuntimeError(f'Invalid scheduler type (SchedulerBase)')

    def __init__(self, optimizer, optimizer_config: OptimizerFactoryBase.Config, scheduler_config: Config):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.counter = 0
        self.scheduler = None

    def step(self, batch_size, t: torch.Tensor=None):
        if batch_size is None:
            batch_size = t.size(dim=0)
        self.counter += batch_size
        if 0 < (self.counter // self.scheduler_config.step_period):
            self.scheduler.step()

    def get_last_lr(self):
        pass

from typing import Optional

import torch
from persistance_helper.SerializableAlg import SerializableAlg
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase


class SchedulerBase(SerializableAlg):

    class Config(object):
        def __init__(self, scheduler_type: str, step_period):
            super().__init__()
            self.scheduler_type = scheduler_type
            self.step_period = step_period

        def get_scheduler_type(self):
            raise RuntimeError(f'Invalid scheduler type (SchedulerBase)')

    class PersistConfig(SerializableAlg.PersistConfig):
        def __init__(self, filename: Optional[str] = None):
            super().__init__(filename=filename if filename is not None else 'scheduler.json')

    def __init__(self, config: Config, optimizer, optimizer_config: OptimizerFactoryBase.Config):
        super().__init__(config=config)
        self.config = config
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config

        self.counter = 0
        self.scheduler = None

    def step(self, batch_size=None, t: torch.Tensor=None):
        if batch_size is None:
            batch_size = t.size(dim=0)
        self.counter += batch_size
        if 0 < (self.counter // self.config.step_period):
            self.scheduler.step()

    def get_last_lr(self):
        pass

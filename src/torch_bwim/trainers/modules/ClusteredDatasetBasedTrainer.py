import torch
from torch.utils.data import DataLoader

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.dataset.batch_samplers.DatasetMapper import DatasetMapper
from torch_bwim.lr_schedulers.SchedulerBase import SchedulerBase
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.trainers.modules.StandardNetTrainer import StandardNetTrainer


class ClusteredDatasetBasedTrainer(StandardNetTrainer):

    class Config(StandardNetTrainer.Config):
        def __init__(self, batch_size, shuffle_dataset, epoch_num=None, random_state=None):
            super().__init__(batch_size, shuffle_dataset, epoch_num=None, random_state=None)

    class State(StandardNetTrainer.State):
        def __init__(self, train_config=None, loss=None,
                     optimizer_config: OptimizerFactoryBase.Config = None,
                     scheduler_config: SchedulerBase.Config = None):
            super().__init__(train_config=train_config, loss=loss,
                             optimizer_config=optimizer_config, scheduler_config=scheduler_config)

    class PersistConfig(StandardNetTrainer.PersistConfig):
        def __init__(self, folder_path=None, train_config_file=None):
            super().__init__(folder_path=folder_path,
                             train_config_file=train_config_file)

    def __init__(self, train_config: Config, dataset_mapper: DatasetMapper, logger=None):
        super().__init__(train_config=train_config, logger=logger)
        self.dataset_mapper = dataset_mapper

    def dataset_to_loader(self, dataset):
        if isinstance(dataset, list):
            dataset, _ = TorchDataUtils.concat_datasets(dataset)
        clustered_dataset, batch_sampler = self.dataset_mapper(
            dataset=dataset, dataset_provider=self.dataset_provider,
            batch_size=self.train_config.batch_size,
            shuffle=self.train_config.shuffle_dataset
        )
        data_loader = torch.utils.data.DataLoader(
            clustered_dataset,
            batch_sampler=batch_sampler
        )
        return data_loader

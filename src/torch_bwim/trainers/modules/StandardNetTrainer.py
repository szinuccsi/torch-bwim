import copy
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.dataset.TrainDictDataset import TrainDictDataset
from torch_bwim.helpers.RandomHelper import RandomHelper
from torch_bwim.lr_schedulers.SchedulerBase import SchedulerBase
from torch_bwim.lr_schedulers.service.SchedulerBuilder import SchedulerBuilder
from torch_bwim.nets.NetBase import NetBase
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.optimizers.service.OptimizerBuilder import OptimizerBuilder
from torch_bwim.trainers.NetTrainerBase import NetTrainerBase
from torch_bwim.trainers.helpers.LearningRatePlotter import LearningRatePlotter
from torch_bwim.trainers.helpers.LossPlotter import LossPlotter
from torch_bwim.trainers.helpers.LossSummary import LossSummary


class StandardNetTrainer(NetTrainerBase):

    class Config(NetTrainerBase.Config):
        def __init__(self, batch_size, shuffle_dataset, epoch_num=None, random_state=None):
            super().__init__(random_state=random_state)
            self.batch_size = batch_size
            self.shuffle_dataset = shuffle_dataset
            self.epoch_num = epoch_num

    def _get_epoch_num(self): return self.train_config.epoch_num

    def _set_epoch_num(self, val):
        if val is not None:
            self.train_config.epoch_num = val
        if self.train_config.epoch_num is None:
            raise RuntimeError(f'train_config.epoch_num is None')

    epoch_num = property(_get_epoch_num, _set_epoch_num)

    class State(NetTrainerBase.State):
        def __init__(self, train_config=None, loss=None,
                     optimizer_config: OptimizerFactoryBase.Config=None,
                     scheduler_config: SchedulerBase.Config=None):
            super().__init__(train_config=train_config, loss=loss)
            self.optimizer_config = optimizer_config
            self.scheduler_config = scheduler_config

    class PersistConfig(NetTrainerBase.PersistConfig):
        def __init__(self, folder_path=None, train_config_file=None):
            super().__init__(folder_path=folder_path,
                             train_config_file=train_config_file)

    def __init__(self, train_config: Config, logger=None):
        super().__init__(train_config=train_config, logger=logger)
        self.train_config: StandardNetTrainer.Config = train_config
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.optimizer_config: OptimizerFactoryBase.Config = None
        self.optimizer = None
        self.scheduler: SchedulerBase = None
        self.best_state: StandardNetTrainer.State = self.State()

        self.dataset_provider: TrainDictDataset = None
        self.loss_function = None
        self.loss_plotter: LossPlotter = None
        self.learning_rate_plotter: LearningRatePlotter = None
        self.cuda = True

    def initialize(self,
                   net: NetBase,
                   train_dataset, val_dataset, dataset_provider: TrainDictDataset,
                   loss_function,
                   scheduler_config: SchedulerBase.Config, optimizer_config: OptimizerFactoryBase.Config,
                   cuda: bool=True,
                   loss_plotter=None, learning_rate_plotter=None):
        self.net = net if not cuda else net.cuda()
        self.train_loader = self.dataset_to_loader(train_dataset)
        self.val_loader = self.dataset_to_loader(val_dataset)
        self.dataset_provider: TrainDictDataset = dataset_provider
        self.loss_function = loss_function
        self.loss_plotter = \
            loss_plotter if loss_plotter is not None else LossPlotter()
        self.learning_rate_plotter = \
            learning_rate_plotter if learning_rate_plotter is not None else LearningRatePlotter()
        self.optimizer_config = optimizer_config
        self.optimizer = OptimizerBuilder.create_optimizer(parameters=self.net.parameters(), config=optimizer_config)
        self.scheduler = SchedulerBuilder.create_scheduler(config=scheduler_config,
                                                           optimizer=self.optimizer, optimizer_config=optimizer_config)
        self.cuda = cuda

    def dataset_to_loader(self, dataset):
        if isinstance(dataset, list):
            dataset, _ = TorchDataUtils.concat_datasets(dataset)
        return DataLoader(dataset,
                          batch_size=self.train_config.batch_size,
                          shuffle=self.train_config.shuffle_dataset,
                          generator=RandomHelper.get_generator(self.train_config.random_state))

    def train(self, epoch_num=None, persist_config: PersistConfig=None,
              plot_required=True):
        super().train()
        self.epoch_num = epoch_num
        for epoch in range(self.epoch_num):
            self.logger(f'Epoch {epoch+1} started at {datetime.datetime.now()}')
            train_loss = self.train_epoch(data_loader=self.train_loader)
            val_loss = self.validate(data_loader=self.val_loader)
            self.loss_plotter.add(train_loss=train_loss, val_loss=val_loss)
            self.best_result(loss=val_loss, persist_config=persist_config)
            self.logger(f'\tTrain loss: {train_loss}')
            self.logger(f'\tValidation loss: {val_loss}')
            self.logger(f'\tLowest loss: {self.best_state.loss}')
        self.loss_plotter.plot(required=plot_required)
        self.learning_rate_plotter.plot(required=plot_required)
        return self.best_state.loss

    def validate(self, data_loader: DataLoader):
        super().validate()
        if isinstance(data_loader, Dataset):
            data_loader = self.dataset_to_loader(data_loader)
        loss_summary = LossSummary()
        for i, data in enumerate(data_loader):
            index = self.dataset_provider.to_index(data, cuda=self.cuda)
            inputs = self.dataset_provider.to_input(data, cuda=self.cuda)
            labels = self.dataset_provider.to_label(data, cuda=self.cuda)
            with torch.no_grad():
                outputs = self.process(inputs=inputs, index=index)
                outputs_and_labels = outputs + labels
                loss = self.loss_function(*outputs_and_labels)
                loss_summary.add(loss.item(), t=index)
        return loss_summary.loss

    def process(self, inputs, index=None):
        outputs = self.net(*inputs)
        return outputs

    def train_epoch(self, data_loader):
        if isinstance(data_loader, Dataset):
            data_loader = self.dataset_to_loader(data_loader)
        self.net.train()
        loss_summary = LossSummary()
        for i, data in enumerate(data_loader):
            index = self.dataset_provider.to_index(data, cuda=self.cuda)
            inputs = self.dataset_provider.to_input(data, cuda=self.cuda)
            labels = self.dataset_provider.to_label(data, cuda=self.cuda)
            self.zero_grad()
            outputs = self.process(inputs=inputs, index=index)
            outputs_and_labels = outputs + labels
            loss = self.loss_function(*outputs_and_labels)
            self.backpropagation(loss=loss)
            self.scheduler.step(t=index)
            self.learning_rate_plotter.add(self.scheduler.get_last_lr())
            loss_summary.add(loss.item(), t=index)
        return loss_summary.loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def best_result(self, loss, persist_config: PersistConfig=None):
        super().best_result(loss=loss, persist_config=persist_config)
        self.best_state.optimizer_config = copy.deepcopy(self.optimizer_config)
        self.best_state.scheduler_config = copy.deepcopy(self.scheduler.config)

    def backpropagation(self, loss):
        loss.backward()
        self.optimizer.step()

    def save(self, persist_config: PersistConfig, with_weights=True):
        if not super().save(persist_config=persist_config, with_weights=with_weights):
            return False
        self.scheduler.save_config(folder_path=persist_config.folder_path)
        OptimizerFactoryBase.save_config(optimizer_config=self.optimizer_config,
                                         folder_path=persist_config.folder_path)
        return True



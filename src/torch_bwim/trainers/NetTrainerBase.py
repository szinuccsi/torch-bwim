import copy

from persistance_helper.PersistHelper import PersistHelper

from torch_bwim.helpers.RandomHelper import RandomHelper
from torch_bwim.nets.NetBase import NetBase


class NetTrainerBase(object):

    class Config(object):
        def __init__(self, random_state=None):
            super().__init__()
            self.random_state = random_state

    class PersistConfig(object):
        def __init__(self, folder_path=None, train_config_file=None):
            super().__init__()
            self.folder_path = folder_path
            self.train_config_file = train_config_file if train_config_file is not None else 'train_config.json'

    class State(object):
        def __init__(self, train_config=None, loss=None):
            super().__init__()
            self.train_config = train_config
            self.loss = loss

        @classmethod
        def is_loss_lower(cls, base_loss, new_loss):
            if base_loss is None:
                return True
            if new_loss < base_loss:
                return True
            return False

    def __init__(self, train_config: Config, logger=None):
        super().__init__()
        self.net: NetBase = None
        self.train_config: NetTrainerBase.Config = train_config
        self.log_config: NetTrainerBase.LogConfig = None
        self.best_state: NetTrainerBase.State = self.State()
        self._logger = logger if logger is not None else lambda s: None

    def initialize(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        RandomHelper.set_random_state(self.train_config.random_state)
        self.net.train()
        self.logger('Train started')

    def validate(self, *args, **kwargs):
        RandomHelper.set_random_state(self.train_config.random_state)
        self.net.eval()
        pass

    def _get_logger(self): return self._logger
    logger = property(_get_logger)

    def best_result(self, loss, persist_config: PersistConfig=None):
        if not self.State.is_loss_lower(base_loss=self.best_state.loss, new_loss=loss):
            return
        self.best_state = self.State(
            loss=loss,
            train_config=copy.deepcopy(self.train_config)
        )
        self.save(persist_config=persist_config)

    def save(self, persist_config: PersistConfig, with_weights=True):
        if (persist_config is None) or (not PersistHelper.valid_path(persist_config.folder_path)):
            return False
        PersistHelper.save_object_to_json(self.train_config, path=PersistHelper.merge_paths([
            persist_config.folder_path, persist_config.train_config_file
        ]))
        self.net.save_net(folder_path=persist_config.folder_path, weights=with_weights)
        return True

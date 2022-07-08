import torch
import torch.nn as nn

from torch_bwim.helpers.PersistHelper import PersistHelper
from torch_bwim.helpers.RandomHelper import RandomHelper
from torch_bwim.helpers.Version import Version


class NetBase(nn.Module):

    class Config(object):
        def __init__(self, version: str=None):
            super().__init__()
            if version is None:
                version = self.get_latest_version()
            if isinstance(version, Version):
                version = str(version)
            self.version = version

        @classmethod
        def get_latest_version(cls):
            return Version(0, 1, 0)

    def __init__(self, config: Config, random_state=None):
        super().__init__()
        self.config = config
        RandomHelper.set_random_state(random_state=random_state)

    class PersistConfig(object):
        def __init__(self, nn_weight_filename='state_dict.pth', nn_config_filename='net_config.json'):
            super().__init__()
            self.nn_weight_filename = nn_weight_filename
            self.nn_config_filename = nn_config_filename

    def save_net(self, folder_path: str, persist_config: PersistConfig=None, weights=True):
        if persist_config is None:
            persist_config = self.PersistConfig()
        self._check_persist_config(folder_path=folder_path, persist_config=persist_config)
        PersistHelper.save_object_to_json(
            data=self.config,
            path=PersistHelper.merge_paths([folder_path, persist_config.nn_config_filename])
        )
        if weights:
            torch.save(
                self.state_dict(),
                PersistHelper.merge_paths([folder_path, persist_config.nn_weight_filename])
            )

    @classmethod
    def load_net(cls, folder_path: str, persist_config: PersistConfig=None, weights=True):
        if persist_config is None:
            persist_config = cls.PersistConfig()
        cls._check_persist_config(folder_path=folder_path, persist_config=persist_config)
        net_config = PersistHelper.load_json_to_object(
            path=PersistHelper.merge_paths([folder_path, persist_config.nn_config_filename])
        )
        net = cls(net_config)
        if weights:
            net.load_state_dict(
                torch.load(
                    PersistHelper.merge_paths([folder_path, persist_config.nn_weight_filename]),
                    map_location='cpu'
                )
            )
        net.eval()
        return net

    @classmethod
    def check_persist_config(cls, folder_path: str, persist_config: PersistConfig):
        if not PersistHelper.valid_path(folder_path):
            raise RuntimeError(f'folder_path({folder_path}) is not valid')
        if not PersistHelper.valid_filename(persist_config.nn_config_filename):
            raise RuntimeError(f'nn_config_filename({persist_config.nn_config_filename}) '
                               f'is not valid')
        if not PersistHelper.valid_filename(persist_config.nn_weight_filename):
            raise RuntimeError(f'nn_weight_filename({persist_config.nn_weight_filename}) '
                               f'is not valid')

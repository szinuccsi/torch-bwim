from torch_bwim.helpers.SerializableAlg import SerializableAlg
from torch_bwim.helpers.Version import Version


class ToTensorAdapter(SerializableAlg):

    class Config(SerializableAlg.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, num_of_features_out, config: Config=None):
        super().__init__(config=config)
        self._num_of_features_out = num_of_features_out

    def _get_num_of_features_out(self):
        return self._num_of_features_out
    num_of_features_out = property(_get_num_of_features_out)

    def __call__(self, data: dict) -> tuple:
        pass

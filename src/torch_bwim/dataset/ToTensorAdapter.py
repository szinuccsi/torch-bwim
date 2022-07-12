from torch_bwim.helpers.SerializableAlg import SerializableAlg
from torch_bwim.helpers.Version import Version


class ToTensorAdapter(SerializableAlg):

    class Config(SerializableAlg.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, length_out, config: Config=None):
        super().__init__(config=config)
        self._length_out = length_out

    def __call__(self, data: dict) -> tuple:
        pass

    def __len__(self):
        return self._length_out

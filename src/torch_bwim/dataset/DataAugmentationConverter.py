from torch_bwim.helpers.SerializableAlg import SerializableAlg
from torch_bwim.helpers.Version import Version


class DataAugmentationConverter(SerializableAlg):

    class Config(SerializableAlg.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, config: Config=None):
        super().__init__(config=config)

    def __call__(self, inputs: tuple, labels: tuple, data: dict):
        pass

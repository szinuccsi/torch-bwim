import torch
from persistance_helper.SerializableAlg import SerializableAlg
from persistance_helper.Version import Version


class LossFunction(SerializableAlg):

    class Config(SerializableAlg.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, config: Config):
        super().__init__(config=config)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

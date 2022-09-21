from typing import Tuple, Union

import torch
from persistance_helper.SerializableAlg import SerializableAlg
from persistance_helper.Version import Version


class DataAugmentationConverter(SerializableAlg):

    class Config(SerializableAlg.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, config: Config=None):
        super().__init__(config=config)

    def process(self, inputs: tuple, labels: tuple, data: dict) -> \
            Tuple[Union[list, tuple, torch.Tensor], Union[list, tuple, torch.Tensor]]:
        pass

    def __call__(self, inputs: tuple, labels: tuple, data: dict) -> Tuple[tuple, tuple]:
        inputs, labels = self.process(inputs=inputs, labels=labels, data=data)
        return self._to_tuple(inputs), self._to_tuple(labels)

    def _to_tuple(self, t):
        if isinstance(t, tuple):
            return t
        if isinstance(t, torch.Tensor):
            return t,
        if isinstance(t, list):
            return tuple(t)
        raise RuntimeError(f'Unknown type')

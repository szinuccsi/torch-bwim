from persistance_helper.SerializableAlg import SerializableAlg
from persistance_helper.Version import Version


class ToTensorAdapter(SerializableAlg):

    class Config(SerializableAlg.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, length_out, config: Config=None):
        super().__init__(config=config)
        self._length_out = length_out

    def process(self, data: dict):
        pass

    def __call__(self, data: dict) -> tuple:
        res = self.process(data)
        if not isinstance(res, tuple):
            res = res,
        return res

    def __len__(self):
        return self._length_out

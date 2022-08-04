from persistance_helper.PersistHelper import PersistHelper


class OptimizerFactoryBase(object):

    class Config(object):
        def __init__(self, optimizer_type: str, learning_rate):
            super().__init__()
            self.optimizer_type = optimizer_type
            self.learning_rate = learning_rate

        @classmethod
        def get_optimizer_type(cls) -> str:
            raise RuntimeError(f'Invalid optimizer_type (AbstractOptimizerFactory)')

    def create(self, parameters, config: Config):
        pass

    class PersistConfig(object):
        def __init__(self, filename: str=None):
            super().__init__()
            self.filename = filename if filename is not None else 'optimizer.json'

    @classmethod
    def save_config(cls, optimizer_config: Config, folder_path, persist_config: PersistConfig=None):
        if persist_config is None:
            persist_config = cls.PersistConfig()
        return PersistHelper.save_object_to_json(
            data=optimizer_config,
            path=PersistHelper.merge_paths([folder_path, persist_config.filename])
        )

    @classmethod
    def load_config(cls, folder_path, persist_config: PersistConfig=None):
        if persist_config is None:
            persist_config = cls.PersistConfig()
        return PersistHelper.load_json_to_object(
            path=PersistHelper.merge_paths([folder_path, persist_config.filename])
        )

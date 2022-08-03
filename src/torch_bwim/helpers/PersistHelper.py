import json
import os
from types import SimpleNamespace


class PersistHelper(object):

    @staticmethod
    def load_json(path: str):
        with open(path) as jsonFile:
            result = json.load(jsonFile)
        return result

    @staticmethod
    def save_json_serializable(data, path: str):
        with open(path, 'w') as jsonSaver:
            json.dump(data, jsonSaver)

    @staticmethod
    def load_json_to_object(path: str):
        loaded_dict = PersistHelper.load_json(path)
        loaded_str = json.dumps(loaded_dict)
        loaded_object = json.loads(loaded_str, object_hook=lambda d: SimpleNamespace(**d))
        return loaded_object

    @staticmethod
    def save_object_to_json(data, path: str):
        with open(path, 'w') as jsonSaver:
            json.dump(data.__dict__, jsonSaver)

    @staticmethod
    def valid_path(path):
        return path is not None and path != ''

    @staticmethod
    def valid_filename(filename):
        return filename is not None and filename != ''

    @staticmethod
    def merge_paths(path_parts):
        return os.path.join(*path_parts)

    @staticmethod
    def create_dir_if_not_exists(dir, log=None):
        if PersistHelper.valid_path(dir):
            if not os.path.isdir(dir):
                if log is not None:
                    log(f'creating {dir}...')
                os.mkdir(dir)
                if log is not None:
                    log(f'{dir} created')

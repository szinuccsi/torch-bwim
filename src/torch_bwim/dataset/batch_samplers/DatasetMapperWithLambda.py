from torch.utils.data import Dataset

from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.batch_samplers.DatasetMapper import DatasetMapper


class DatasetMapperWithLambda(DatasetMapper):

    def __init__(self, key_of_sample):
        super().__init__()
        self._key_of_sample = key_of_sample

    def key_of_sample(self, dataset: Dataset, full_dataset: DictDataset, index):
        return self._key_of_sample(dataset, full_dataset, index)

from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.batch_samplers.DatasetMapper import DatasetMapper


class FeatureBasedMapper(DatasetMapper):

    def __init__(self, Feature):
        super().__init__()
        self.Feature = Feature

    def key_of_sample(self, dataset: Dataset, full_dataset: DictDataset, index):
        data: dict = full_dataset.get_sample(index)
        return data[self.Feature]

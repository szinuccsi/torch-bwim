from torch.utils.data import Dataset

from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.dataset.batch_samplers.ClusterBasedBatchSampler import ClusterBasedBatchSampler


class DatasetMapper(object):

    def __init__(self):
        super().__init__()

    def __call__(self, dataset: Dataset, full_dataset: DictDataset, batch_size, shuffle):
        clustered_datasets = self.map_samples(dataset, full_dataset)
        clustered_datasets = TorchDataUtils.concat_datasets_in_dict(clustered_dataset)
        batch_sampler = ClusterBasedBatchSampler(clustered_datasets=clustered_datasets,
                                                 batch_size=batch_size, shuffle=shuffle)
        return clustered_dataset, batch_sampler

    def map_samples(self, dataset: Dataset, full_dataset: DictDataset):
        clustered_dataset = {}
        for data in dataset:
            index = int(data[0])
            key = self.key_of_sample(dataset=dataset, full_dataset=full_dataset, index=index)
            HelperFunctions.add_element_to_array_in_dict(clustered_dataset, key=key, element=data)
        return clustered_dataset

    def key_of_sample(self, dataset: Dataset, full_dataset: DictDataset, index):
        pass

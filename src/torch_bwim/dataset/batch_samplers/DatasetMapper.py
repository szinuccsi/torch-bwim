from torch.utils.data import Dataset

from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.dataset.batch_samplers.ClusterBasedBatchSampler import ClusterBasedBatchSampler


class DatasetMapper(object):

    def __init__(self):
        super().__init__()

    def __call__(self, dataset: Dataset, full_dataset: DictDataset, batch_size, shuffle, random_state=None):
        clustered_datasets = self.map_samples(dataset, full_dataset)
        concat_clustered_datasets, cluster_sizes = TorchDataUtils.concat_datasets_in_dict(clustered_datasets)
        batch_sampler = ClusterBasedBatchSampler(cluster_sizes=cluster_sizes,
                                                 batch_size=batch_size, shuffle=shuffle, random_state=random_state)
        return concat_clustered_datasets, batch_sampler

    def map_samples(self, dataset: Dataset, full_dataset: DictDataset):
        clustered_dataset = {}
        for data in dataset:
            index = int(data[0])
            key = self.key_of_sample(dataset=dataset, full_dataset=full_dataset, index=index)
            self._add_element_to_array_in_dict(clustered_dataset, key=key, element=data)
        return clustered_dataset

    def key_of_sample(self, dataset: Dataset, full_dataset: DictDataset, index):
        pass

    def _add_element_to_array_in_dict(self, d: dict, key, element):
        if d.get(key) is None:
            d[key] = []
        d[key].append(element)

import random
import torch.utils.data

from torch_bwim.dataset.TorchDataUtils import TorchDataUtils
from torch_bwim.helpers.RandomHelper import RandomHelper


class ClusterBasedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, cluster_sizes, batch_size: int, shuffle, random_state=None):
        super().__init__(data_source=None)
        RandomHelper.set_random_state(random_state)
        self.cluster_sizes = cluster_sizes

        self.batches = []
        start_idx = 0
        for cluster in range(len(self.cluster_sizes)):
            end_idx = start_idx + self.cluster_sizes[cluster]
            cluster_indices = [idx for idx in range(start_idx, end_idx)]
            if shuffle:
                random.shuffle(cluster_indices)
            self.batches.extend(self.batches_create(cluster_indices, batch_size))
            start_idx = end_idx
        self.batch_indices = [batch_idx for batch_idx in range(len(self.batches))]
        if shuffle:
            random.shuffle(self.batch_indices)

    def batches_create(self, indices, batch_size):
        batches = []
        length = len(indices)
        for i in range(0, length, batch_size):
            batches.append(indices[i:min(i + batch_size, length)])
        return batches

    def __iter__(self):
        for index in range(len(self.batch_indices)):
            batch = self.batches[self.batch_indices[index]]
            yield batch

    def __len__(self):
        return len(self.batches)

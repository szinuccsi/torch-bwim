import torch
import torch.utils.data
from torch.utils.data import Dataset

from torch_bwim.helpers.RandomHelper import RandomHelper


class TorchDataUtils(object):

    @classmethod
    def check_shape(cls, t: torch.tensor, expected_shape, tensor_name='tensor', throw_error=True):
        if len(t.shape) != len(expected_shape):
            if throw_error:
                raise RuntimeError(f'{tensor_name} shape len is {len(t.shape)} , '
                               f'expected {len(expected_shape)}')
            else:
                return False
        n = len(t.shape)
        for i in range(n):
            if (expected_shape[i] is not None) and (t.shape[i] != expected_shape[i]):
                if throw_error:
                    raise RuntimeError(f'{tensor_name} shape is {t.shape}, expected {expected_shape}')
                else:
                    return False
        return True

    @classmethod
    def concat_datasets(cls, clustered_datasets):
        subset_lens = []
        datasets_to_concat = []
        if isinstance(clustered_datasets, dict):
            for key in clustered_datasets:
                d = clustered_datasets[key]
                subset_lens.append(len(d))
                datasets_to_concat.append(d)
            concat_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)
            return concat_dataset, subset_lens
        elif isinstance(clustered_datasets, list):
            for d in clustered_datasets:
                subset_lens.append(len(d))
                datasets_to_concat.append(d)
            concat_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)
            return concat_dataset, subset_lens
        else:
            raise RuntimeError(f'clustered_dataset type is not list or dict ({type(clustered_datasets)})')

    @classmethod
    def split_dataset(cls, dataset, length_ratios=[], random_state=None):
        RandomHelper.set_random_state(random_state)
        if len(length_ratios) == 0:
            return dataset
        if sum(length_ratios) != 1:
            raise RuntimeError('length_ratios sum is not 1: got {sum}'.format(
                sum=sum(length_ratios))
            )
        lengths = [int(len(dataset) * ratio) for ratio in length_ratios]
        lengths[-1] += len(dataset) - sum(lengths)
        return torch.utils.data.random_split(dataset=dataset, lengths=lengths,
                                             generator=RandomHelper.get_generator(random_state=random_state))

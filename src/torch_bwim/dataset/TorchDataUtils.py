import torch
import torch.utils.data
from torch.utils.data import Dataset


class TorchDataUtils(object):

    @classmethod
    def unsqueeze_tensors(cls, tensors, dim=0):
        res = []
        for t in tensors:
            new_t = t.unsqueeze(dim)
            res.append(new_t)
        if isinstance(tensors, tuple):
            res = tuple(res)
        return res

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
    def cuda_is_available(cls, cuda=True):
        return torch.cuda.is_available() and cuda

    @classmethod
    def to_device(cls, t: torch.Tensor, cuda=True):
        if (not isinstance(t, list)) and (not isinstance(t, tuple)):
            return t.cuda() if cls.cuda_is_available() else t
        if not isinstance(cuda, list):
            cuda = [cuda for _ in range(len(t))]
        if len(cuda) != len(t):
            raise RuntimeError(f'len(cuda)({len(cuda)}) != len(t)({len(t)})')
        t = [cls.to_device(t[i], cuda[i]) for i in range(len(t))]
        if isinstance(t, tuple):
            return tuple(t)
        return t

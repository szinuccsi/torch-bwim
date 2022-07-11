import torch


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
    def check_shape(cls, t: torch.tensor, expected_shape, tensor_name='tensor'):
        if len(t.shape) != len(expected_shape):
            raise RuntimeError(f'{tensor_name} shape len is {len(t.shape)} , '
                               f'expected {len(expected_shape)}')
        n = len(t.shape)
        for i in range(n):
            if (expected_shape[i] is not None) and (t.shape[i] != expected_shape[i]):
                raise RuntimeError(f'{tensor_name} shape is {t.shape}, expected {expected_shape}')

    @classmethod
    def concat_datasets_in_dict(cls, clustered_dataset: dict):
        cluster_sizes = []
        datasets_to_concat = []
        for key in clustered_dataset:
            cluster_sizes.append(len(clustered_dataset[key]))
            datasets_to_concat.append(clustered_dataset[key])
        concat_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)
        return concat_dataset, cluster_sizes

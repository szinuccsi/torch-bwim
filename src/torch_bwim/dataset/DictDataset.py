from torch_bwim.dataset.ToTensorAdapter import ToTensorAdapter
from torch_bwim.dataset.TorchDataUtils import TorchDataUtils


class DictDataset(object):

    def __init__(self, dict_dataset,
                 input_to_tensor_adapter: ToTensorAdapter):
        super().__init__()
        self._dict_dataset = dict_dataset
        self._input_feats_range = None
        self._input_to_tensor_adapter: ToTensorAdapter = input_to_tensor_adapter
        if len(self) == 0:
            raise RuntimeError(f'len(self.num_of_samples)({len(self)}) == 0')

    def __len__(self):
        return len(self._dict_dataset)

    def get_sample(self, index):
        index %= len(self)
        return self._dict_dataset[index]

    def _get_input(self, index):
        inputs = self._input_to_tensor_adapter(self.get_sample(index))
        self._input_feats_range = (1, 1+len(inputs))
        return inputs

    def __getitem__(self, index):
        inputs = self._get_input(index)
        res = (index,) + inputs
        return res

    def to_index(self, tensors, cuda=False):
        return TorchDataUtils.to_device(tensors[0], cuda=cuda)

    def to_input(self, tensors, cuda=False):
        return TorchDataUtils.to_device(tensors[self.input_feats_range[0]:self.input_feats_range[1]], cuda=cuda)

    def get_input(self, index):
        res = self.__getitem__(index)
        return self.to_input(res)

    def _get_input_feats_range(self):
        if self._input_feats_range is None:
            self._input_feats_range = (1, 1+len(self._input_to_tensor_adapter))
        return self._input_feats_range
    input_feats_range = property(_get_input_feats_range)

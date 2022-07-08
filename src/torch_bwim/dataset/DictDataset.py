from torch_bwim.dataset.ToTensorAdapter import ToTensorAdapter


class DictDataset(object):

    def __init__(self, dict_dataset,
                 input_to_tensor_adapter: ToTensorAdapter):
        super().__init__()
        self._dict_dataset = dict_dataset

        self._input_to_tensor_adapter: ToTensorAdapter = input_to_tensor_adapter
        self._input_feats_range = (1, 1 + input_to_tensor_adapter.num_of_features_out)

        if len(self) == 0:
            raise RuntimeError(f'len(self.num_of_samples)({len(self)}) == 0')

    def __len__(self):
        return len(self._dict_dataset)

    def get_sample(self, index):
        index %= len(self)
        return self._dict_dataset[index]

    def get_input(self, index):
        inputs = self._input_to_tensor_adapter(self.get_sample(index))
        return inputs

    def __getitem__(self, index):
        inputs = self.get_input(index)
        res = (index,) + inputs
        return res

    def _get_input_feats_range(self):
        return self._input_feats_range
    input_feats_range = property(_get_input_feats_range)

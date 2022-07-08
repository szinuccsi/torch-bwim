from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.ToTensorAdapter import ToTensorAdapter


class TrainDictDataset(DictDataset):

    def __init__(self, dict_dataset,
                 input_to_tensor_adapter: ToTensorAdapter,
                 label_to_tensor_adapter: ToTensorAdapter,
                 data_augmentation_converter=None):
        super().__init__(dict_dataset=dict_dataset, input_to_tensor_adapter=input_to_tensor_adapter)
        self._label_to_tensor_adapter: ToTensorAdapter = label_to_tensor_adapter
        self._label_feats_range = (self._input_feats_range[1],
                                   self.input_feats_range[1] + label_to_tensor_adapter.num_of_features_out)
        self.data_augmentation_converter = data_augmentation_converter

    def get_label(self, index):
        return self._label_to_tensor_adapter(self.get_sample(index))

    def __getitem__(self, index):
        inputs = self.get_input(index)
        labels = self.get_label(index)
        if self.data_augmentation_converter is not None:
            inputs, labels = self.data_augmentation_converter(inputs, labels, data=self.get_sample(index))
        res = (index,) + inputs + labels
        return res

    def _get_label_feats_range(self):
        return self._label_feats_range
    label_feats_range = property(_get_label_feats_range)

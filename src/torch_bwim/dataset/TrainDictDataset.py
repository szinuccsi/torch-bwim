from torch_bwim.dataset.DataAugmentationConverter import DataAugmentationConverter
from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.ToTensorAdapter import ToTensorAdapter
from torch_bwim.dataset.TorchDataUtils import TorchDataUtils


class TrainDictDataset(DictDataset):

    def __init__(self, dict_dataset,
                 input_to_tensor_adapter: ToTensorAdapter,
                 label_to_tensor_adapter: ToTensorAdapter,
                 data_augmentation_converter: DataAugmentationConverter=None):
        super().__init__(dict_dataset=dict_dataset, input_to_tensor_adapter=input_to_tensor_adapter)
        self._label_to_tensor_adapter: ToTensorAdapter = label_to_tensor_adapter

        self._label_feats_range = None
        self.data_augmentation_converter = data_augmentation_converter

    def _get_label(self, index):
        labels = self._label_to_tensor_adapter(self.get_sample(index))
        self._label_feats_range = (self.input_feats_range[1], self.input_feats_range[1] + len(labels))
        return labels

    def __getitem__(self, index):
        inputs = self._get_input(index)
        labels = self._get_label(index)
        if self.data_augmentation_converter is not None:
            inputs, labels = self.data_augmentation_converter(inputs, labels, data=self.get_sample(index))
        res = (index,) + inputs + labels
        return res

    def to_label(self, tensors, cuda=False):
        return TorchDataUtils.to_device(tensors[self.label_feats_range[0]: self.label_feats_range[1]], cuda=cuda)

    def get_label(self, index):
        res = self.__getitem__(index)
        return self.to_label(res)

    def _get_label_feats_range(self):
        if self._label_feats_range is None:
            input_feats_range = self.input_feats_range
            self._label_feats_range = (input_feats_range[1],
                                       input_feats_range[1] + len(self._label_to_tensor_adapter))
        return self._label_feats_range
    label_feats_range = property(_get_label_feats_range)

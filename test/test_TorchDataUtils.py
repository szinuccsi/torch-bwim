import unittest
import torch

from torch_bwim.dataset.DictDataset import DictDataset
from torch_bwim.dataset.TorchDataUtils import TorchDataUtils


class TorchDataUtilsTestCase(unittest.TestCase):

    def test_check_shape_pass(self):
        t = torch.randn((128, 48, 32))
        self.assertTrue(TorchDataUtils.check_shape(t=t, expected_shape=(128, 48, 32)))
        self.assertTrue(TorchDataUtils.check_shape(t=t, expected_shape=(128, None, 32)))
        self.assertTrue(TorchDataUtils.check_shape(t=t, expected_shape=(None, 48, None)))

    def test_check_shape_fail(self):
        t = torch.randn((128, 48, 32))

        self.assertFalse(TorchDataUtils.check_shape(t=t, expected_shape=(None, None, None, None), throw_error=False))
        self.assertFalse(TorchDataUtils.check_shape(t=t, expected_shape=(None, None), throw_error=False))
        self.assertFalse(TorchDataUtils.check_shape(t=t, expected_shape=(None, 48, 0), throw_error=False))
        self.assertFalse(TorchDataUtils.check_shape(t=t, expected_shape=(100, 48, 32), throw_error=False))
        self.assertFalse(TorchDataUtils.check_shape(t=t, expected_shape=(None, None, 0), throw_error=False))
        self.assertFalse(TorchDataUtils.check_shape(t=t, expected_shape=(128, 0, 0), throw_error=False))

    def test_unsqueeze_tensors(self):
        t = torch.randn((128, 48, 32))
        u = torch.randn((16, 8))

        [res_0] = TorchDataUtils.unsqueeze_tensors([t], dim=0)
        self.assertTrue(TorchDataUtils.check_shape(res_0, expected_shape=(1, 128, 48, 32)))

        [res_10, res_11] = TorchDataUtils.unsqueeze_tensors([t, u], dim=1)
        self.assertTrue(TorchDataUtils.check_shape(res_10, expected_shape=(128, 1, 48, 32)))
        self.assertTrue(TorchDataUtils.check_shape(res_11, expected_shape=(16, 1, 8)))

    DATASET1 = DictDataset(dict_dataset=[0, 1, 2],
                           input_to_tensor_adapter=lambda d: (d, d))
    DATASET2 = DictDataset(dict_dataset=[0, 1, 2, 3, 4],
                           input_to_tensor_adapter=lambda d: (d, d))
    DATASET3 = DictDataset(dict_dataset=[5, 6, 7, 8],
                           input_to_tensor_adapter=lambda d: (d, d))

    CONCAT_LEN = len(DATASET1) + len(DATASET2) + len(DATASET3)

    def load_list_of_datasets(self):
        return [self.DATASET1, self.DATASET2, self.DATASET3]

    def load_dict_of_datasets(self):
        return {
            1: self.DATASET1,
            2: self.DATASET2,
            3: self.DATASET3
        }

    def test_concat_datasets_list(self):
        concat_dataset, subset_lens = TorchDataUtils.concat_datasets(self.load_list_of_datasets())
        self.assertEqual(len(concat_dataset), self.CONCAT_LEN)
        self.assertEqual(subset_lens[0], len(self.DATASET1))
        self.assertEqual(subset_lens[1], len(self.DATASET2))
        self.assertEqual(subset_lens[2], len(self.DATASET3))

    def test_concat_datasets_dict(self):
        concat_dataset, subset_lens = TorchDataUtils.concat_datasets(self.load_dict_of_datasets())
        self.assertEqual(len(concat_dataset), self.CONCAT_LEN)
        self.assertEqual(subset_lens[0], len(self.DATASET1))
        self.assertEqual(subset_lens[1], len(self.DATASET2))
        self.assertEqual(subset_lens[2], len(self.DATASET3))


if __name__ == '__main__':
    unittest.main()

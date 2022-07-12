import unittest

from torch_bwim.dataset.DictDataset import DictDataset


class DictDatasetTestCase(unittest.TestCase):

    ATTR = 'attr'

    FIRST = 'first'
    SECOND = 'second'
    THIRD = 'third'

    INPUT_TUPLE_LEN = 2
    INPUT_TUPLE_WITH_INDEX_LEN = 3

    A_OBJ = {ATTR: FIRST}
    B_OBJ = {ATTR: SECOND}
    C_OBJ = {ATTR: THIRD}

    PLUS_INFO = 'plus_info'

    @classmethod
    def load_data(cls):
        return [
            cls.A_OBJ,
            cls.B_OBJ,
            cls.C_OBJ
        ]

    @classmethod
    def load_dataset(cls):
        return DictDataset(
            dict_dataset=cls.load_data(),
            input_to_tensor_adapter=lambda d: (d[cls.ATTR], cls.PLUS_INFO)
        )

    def test_dataset_len(self):
        dict_list = self.load_data()
        dataset = self.load_dataset()
        self.assertTrue(len(dict_list), len(dataset))

    def test_output_len(self):
        dict_list = self.load_data()
        dataset = self.load_dataset()

        for i in range(2 * len(dict_list)):
            self.assertEqual(len(dataset[i]), self.INPUT_TUPLE_WITH_INDEX_LEN)
            self.assertEqual(dataset.input_feats_range[0], 1)
            self.assertEqual(dataset.input_feats_range[1], self.INPUT_TUPLE_WITH_INDEX_LEN)

    def test_get_sample(self):
        dataset = self.load_dataset()
        self.assertTrue(dataset.get_sample(0) is self.A_OBJ)
        self.assertTrue(dataset.get_sample(1) is self.B_OBJ)
        self.assertTrue(dataset.get_sample(2) is self.C_OBJ)
        self.assertTrue(dataset.get_sample(3) is self.A_OBJ)

    def test_input_len(self):
        dataset = self.load_dataset()

        self.assertEqual(len(dataset.get_input(0)), self.INPUT_TUPLE_LEN)

    def test_input(self):
        dataset = self.load_dataset()

        i_50, i_51 = dataset.get_input(5)

        self.assertEqual(i_50, self.THIRD)
        self.assertEqual(i_51, self.PLUS_INFO)

    def test_output(self):
        dataset = self.load_dataset()

        idx_0, i_00, i_01 = dataset[0]
        idx_1, i_10, i_11 = dataset[1]

        self.assertEqual(idx_0, 0)
        self.assertEqual(i_00, self.FIRST)
        self.assertEqual(i_01, self.PLUS_INFO)

        self.assertEqual(idx_1, 1)
        self.assertEqual(i_10, self.SECOND)
        self.assertEqual(i_11, self.PLUS_INFO)


if __name__ == '__main__':
    unittest.main()

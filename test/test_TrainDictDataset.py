import unittest

from torch_bwim.dataset.TrainDictDataset import TrainDictDataset


class TrainDictDatasetTestCase(unittest.TestCase):

    WORD_1 = 'alma'
    WORD_2 = 'beka'
    WORD_3 = 'cekla'

    INPUT_WORD = 'input'
    LABEL_WORD = 'label'

    LABEL_LEN = 3
    LABEL_RANGE = (1+2, 1+2+3)

    OUTPUT_LEN = 1+2+3

    @classmethod
    def load_data(cls):
        return [
            cls.WORD_1,
            cls.WORD_2,
            cls.WORD_3
        ]

    @classmethod
    def load_dataset(cls):
        return TrainDictDataset(
            dict_dataset=cls.load_data(),
            input_to_tensor_adapter=lambda w: (f'{w}_{cls.INPUT_WORD}', f'{cls.INPUT_WORD}_{w}'),
            label_to_tensor_adapter=lambda w: (w, f'{w}_{cls.LABEL_WORD}', f'{cls.LABEL_WORD}_{w}')
        )

    def test_label_len(self):
        word_list = self.load_data()
        dataset = self.load_dataset()

        for i in range(2 * len(word_list)):
            dataset.get_input(i)
            labels = dataset.get_label(i)
            self.assertEqual(len(labels), self.LABEL_LEN)
            self.assertEqual(dataset.label_feats_range[0], self.LABEL_RANGE[0])
            self.assertEqual(dataset.label_feats_range[1], self.LABEL_RANGE[1])

    def test_label(self):
        dataset = self.load_dataset()

        l_01, l_02, l_03 = dataset.get_label(0)
        l_21, l_22, l_23 = dataset.get_label(2)

        self.assertEqual(l_01, self.WORD_1)
        self.assertEqual(l_02, f'{self.WORD_1}_{self.LABEL_WORD}')
        self.assertEqual(l_03, f'{self.LABEL_WORD}_{self.WORD_1}')

        self.assertEqual(l_21, self.WORD_3)
        self.assertEqual(l_22, f'{self.WORD_3}_{self.LABEL_WORD}')
        self.assertEqual(l_23, f'{self.LABEL_WORD}_{self.WORD_3}')

    def test_output_len(self):
        dataset = self.load_dataset()

        item = dataset[0]

        self.assertEqual(len(item), self.OUTPUT_LEN)

    def test_output(self):
        dataset = self.load_dataset()

        idx, i1, i2, l1, l2, l3 = dataset[4]

        self.assertEqual(idx, 4)
        self.assertEqual(i1, f'{self.WORD_2}_{self.INPUT_WORD}')
        self.assertEqual(i2, f'{self.INPUT_WORD}_{self.WORD_2}')
        self.assertEqual(l1, self.WORD_2)
        self.assertEqual(l2, f'{self.WORD_2}_{self.LABEL_WORD}')
        self.assertEqual(l3, f'{self.LABEL_WORD}_{self.WORD_2}')


if __name__ == '__main__':
    unittest.main()

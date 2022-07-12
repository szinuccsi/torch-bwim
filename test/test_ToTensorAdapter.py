import unittest

from torch_bwim.dataset.ToTensorAdapter import ToTensorAdapter


class ToTensorAdapterTestCase(unittest.TestCase):

    def test_len(self):
        to_tensor = ToTensorAdapter(length_out=4)

        self.assertEqual(len(to_tensor), 4)


if __name__ == '__main__':
    unittest.main()

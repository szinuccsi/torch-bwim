import unittest

from torch_bwim.dataset.ToTensorAdapter import ToTensorAdapter


class ToTensorAdapterTestCase(unittest.TestCase):

    class OneOutMock(ToTensorAdapter):
        def __init__(self, key):
            super().__init__(length_out=None, config=None)
            self.key = key

        def process(self, data: dict):
            return data[self.key]

    class MoreOutMock(ToTensorAdapter):
        def __init__(self):
            super().__init__(length_out=None, config=None)

        OBJ1 = 'obj1'
        OBJ2 = 'obj2'

        def process(self, data: dict):
            return self.OBJ1, self.OBJ2

    def test_len(self):
        to_tensor = ToTensorAdapter(length_out=4)

        self.assertEqual(len(to_tensor), 4)

    KEY = 'key'
    OBJ = 'obj'

    def test_one_output_tuple(self):
        to_tensor = self.OneOutMock(key=self.KEY)
        res = to_tensor({self.KEY: self.OBJ})

        self.assertTrue(isinstance(res, tuple))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], self.OBJ)

    def test_more_output_tuple(self):
        to_tensor = self.MoreOutMock()
        res = to_tensor({})

        self.assertTrue(isinstance(res, tuple))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], self.MoreOutMock.OBJ1)
        self.assertEqual(res[1], self.MoreOutMock.OBJ2)


if __name__ == '__main__':
    unittest.main()

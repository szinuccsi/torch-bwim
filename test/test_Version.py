import unittest

from torch_bwim.helpers.Version import Version


class VersionTestCase(unittest.TestCase):

    def test_to_str(self):
        version = Version(major=1, minor=2, patch=3)

        self.assertEqual(str(version), '1.2.3')

    def test_convert_str(self):
        version_str = '2.3.6'
        version = Version.convert_str(version_str=version_str)
        self.assertEqual(version.major, 2)
        self.assertEqual(version.minor, 3)
        self.assertEqual(version.patch, 6)

    def test_eq(self):
        a = Version(2, 3, 5)
        b = Version(2, 3, 5)
        self.assertTrue(a == b)

    def test_neq(self):
        self.assertTrue(Version(1, 3, 5) != Version(2, 3, 5))
        self.assertTrue(Version(2, 4, 5) != Version(2, 3, 5))
        self.assertTrue(Version(2, 3, 6) != Version(2, 3, 5))
        self.assertTrue(Version(1, 1, 1) != Version(2, 3, 5))

    def test_lt(self):
        self.assertTrue(Version(1, 6, 6) < Version(2, 3, 3))
        self.assertTrue(Version(2, 1, 6) < Version(2, 3, 3))
        self.assertTrue(Version(2, 3, 1) < Version(2, 3, 3))

    def test_nlt(self):
        self.assertFalse(Version(2, 1, 1) < Version(1, 2, 2))

    def test_le(self):
        self.assertTrue(Version(1, 1, 1) <= Version(1, 2, 2))
        self.assertTrue(Version(1, 1, 1) <= Version(1, 1, 1))


if __name__ == '__main__':
    unittest.main()

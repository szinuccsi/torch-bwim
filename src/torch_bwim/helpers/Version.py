import re


class Version(object):

    def __init__(self, major, minor, patch=0):
        super().__init__()
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self):
        return f'{self.major}.{self.minor}.{self.patch}'

    def __cmp__(self, other):
        if not isinstance(other, Version):
            raise RuntimeError(f'')
        self_parts = [self.major, self.minor, self.patch]
        other_parts = [other.major, other.minor, other.patch]
        for i in range(len(self_parts)):
            diff = self_parts[i] - other_parts[i]
            if diff != 0:
                return diff
        return 0

    def __eq__(self, other): return self.__cmp__(other) == 0

    def __ne__(self, other): return self.__cmp__(other) != 0

    def __gt__(self, other): return self.__cmp__(other) > 0

    def __lt__(self, other): return self.__cmp__(other) < 0

    def __ge__(self, other): return self.__cmp__(other) >= 0

    def __le__(self, other): return self.__cmp__(other) <= 0

    @classmethod
    def convert_str(cls, version_str: str):
        matches = re.findall(r"^[0-9]+\.[0-9]+\.[0-9]$", version_str)
        print(matches)
        if len(matches) != 1:
            raise RuntimeError(f'Invalid version_str: {version_str}')
        version_parts_str = version_str.split('.')
        [major, minor, patch] = [int(s) for s in version_parts_str]
        return Version(major, minor, patch)

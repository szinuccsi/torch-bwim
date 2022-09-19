from torch.utils.data import DataLoader, Dataset


class TestDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, shuffle: bool = False):
        super().__init__(dataset=dataset, batch_size=1, shuffle=shuffle)

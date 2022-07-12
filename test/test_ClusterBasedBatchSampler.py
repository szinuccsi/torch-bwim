import unittest

from torch_bwim.dataset.batch_samplers.ClusterBasedBatchSampler import ClusterBasedBatchSampler


class ClusterBasedBatchSamplerTestCase(unittest.TestCase):

    CLUSTER_SIZES = [5, 4, 9, 1, 12]
    BATCH_SIZE = 4

    BATCHES = [
        [i for i in range(0, 4)],
        [i for i in range(4, 5)],
        [i for i in range(5, 9)],
        [i for i in range(9, 13)],
        [i for i in range(13, 17)],
        [i for i in range(17, 18)],
        [i for i in range(18, 19)],
        [i for i in range(19, 23)],
        [i for i in range(23, 27)],
        [i for i in range(27, 31)],
    ]

    def test_batches_with_cluster_sizes(self):
        cluster_based_batch_sampler = ClusterBasedBatchSampler(cluster_sizes=self.CLUSTER_SIZES,
                                                               batch_size=self.BATCH_SIZE, shuffle=False)
        self.assertEqual(len(cluster_based_batch_sampler), len(self.BATCHES))
        for i, b in enumerate(cluster_based_batch_sampler):
            self.assertEqual(len(b), len(self.BATCHES[i]))
            for j in range(len(b)):
                self.assertEqual(b[j], self.BATCHES[i][j])


if __name__ == '__main__':
    unittest.main()

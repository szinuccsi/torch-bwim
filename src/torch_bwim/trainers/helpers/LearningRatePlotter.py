import matplotlib.pyplot as plt


class LearningRatePlotter(object):

    def __init__(self):
        super().__init__()
        self.lrs = []

    def add(self, learning_rate):
        self.lrs.append(learning_rate)

    def clear(self):
        self.lrs = []

    def plot(self, log_scale=True, required=True):
        if not required:
            return
        plt.plot([i for i in range(len(self.lrs))], self.lrs)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Lr')
        plt.legend()
        plt.tight_layout()
        plt.show()

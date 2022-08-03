import matplotlib.pyplot as plt


class LossPlotter(object):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def add(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def clear(self):
        self.train_losses = []
        self.val_losses = []

    def plot(self, log_scale=True, required=True):
        if not required:
            return False
        plt.plot([i for i in range(1, len(self.train_losses)+1)], self.train_losses, label='train')
        plt.plot([i for i in range(1, len(self.train_losses)+1)], self.val_losses, label='val')
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()


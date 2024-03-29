import random
import torch
import numpy as np


class RandomHelper(object):

    @classmethod
    def set_random_state(cls, random_state):
        if random_state is not None:
            torch.manual_seed(random_state)
            random.seed(random_state)
            np.random.seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

    @classmethod
    def get_generator(cls, random_state=None):
        generator = torch.Generator()
        if random_state is not None:
            generator.manual_seed(random_state)
        return generator

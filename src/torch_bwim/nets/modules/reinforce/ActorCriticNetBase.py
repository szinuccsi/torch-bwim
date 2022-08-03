from collections import namedtuple

import torch

from torch_bwim.helpers.Version import Version
from torch_bwim.nets.NetBase import NetBase


class ActorCriticNetBase(NetBase):

    class Config(NetBase.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._history = []
        self._exploration: bool = False

    ReplayObject = namedtuple('ReplayObject', ('log_prob', 'value'))

    def reset(self):
        del self._history[:]
        self._history = []

    def _get_exploration(self): return self._exploration
    def _set_exploration(self, val: bool): self._exploration = val
    exploration = property(_get_exploration, _set_exploration)

    def __call__(self, state, exploration=None):
        if not isinstance(state, tuple):
            state = state,
        action_scores, baseline = self.forward(*state)
        action = self.select_action(action_scores=action_scores, baseline=baseline, exploration=exploration)
        return action

    def select_action(self, action_scores, baseline, exploration):
        pass

    def _get_history(self): return self._history
    history = property(_get_history)

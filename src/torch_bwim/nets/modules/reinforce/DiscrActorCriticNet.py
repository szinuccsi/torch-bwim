from persistance_helper.Version import Version
from torch.distributions import Categorical

from torch_bwim.nets.modules.reinforce.ActorCriticNetBase import ActorCriticNetBase


class DiscrActorCriticNet(ActorCriticNetBase):

    class Config(ActorCriticNetBase.Config):
        def __init__(self, version: Version=None):
            super().__init__(version=version)

    def __init__(self, config: Config):
        super().__init__(config=config)

    def select_action(self, action_scores, baseline, exploration):
        distr = Categorical(action_scores)
        action = distr.sample()
        self.history.append(self.ReplayObject(distr.log_prob(action), baseline))
        # TODO exploration = True
        return action

import gym

from torch_bwim.lr_schedulers.SchedulerBase import SchedulerBase
from torch_bwim.lr_schedulers.service.SchedulerBuilder import SchedulerBuilder
from torch_bwim.nets.NetBase import NetBase
from torch_bwim.nets.modules.reinforce.ActorCriticNetBase import ActorCriticNetBase
from torch_bwim.optimizers.OptimizerFactoryBase import OptimizerFactoryBase
from torch_bwim.optimizers.service.OptimizerBuilder import OptimizerBuilder
from torch_bwim.trainers.NetTrainerBase import NetTrainerBase
from torch_bwim.trainers.helpers.LearningRatePlotter import LearningRatePlotter
from torch_bwim.trainers.helpers.LossPlotter import LossPlotter


class ActorCriticNetTrainer(NetTrainerBase):

    class Config(NetTrainerBase.Config):
        def __init__(self, episode_num=None, max_iter_in_episode=None, random_state=None):
            super(self).__init__(random_state=random_state)
            self.episode_num = episode_num
            self.max_iter_in_episode = None

    def __init__(self, train_config: Config, logger=None):
        super().__init__(train_config=train_config, logger=logger)
        self.train_config = train_config
        self.net: ActorCriticNetBase = None
        self.env: gym.Env = None
        self.loss_function = None
        self.learning_rate_plotter = None
        self.loss_plotter = None
        self.optimizer_config = None
        self.optimizer = None
        self.scheduler = None
        self.cuda = None

        self.rewards = []

    def initialize(self, net: ActorCriticNetBase,
                   env: gym.Env, loss_function,
                   scheduler_config: SchedulerBase.Config, optimizer_config: OptimizerFactoryBase.Config,
                   cuda=True,
                   loss_plotter=None, learning_rate_plotter=None):
        self.net = net
        self.env = env
        self.loss_function = loss_function
        self.loss_plotter = \
            loss_plotter if loss_plotter is not None else LossPlotter()
        self.learning_rate_plotter = \
            learning_rate_plotter if learning_rate_plotter is not None else LearningRatePlotter()
        self.optimizer_config = optimizer_config
        self.optimizer = OptimizerBuilder.create_optimizer(parameters=self.net.parameters(), config=optimizer_config)
        self.scheduler = SchedulerBuilder.create_scheduler(config=scheduler_config,
                                                           optimizer=self.optimizer, optimizer_config=optimizer_config)
        self.cuda = cuda

    def _get_episode_num(self): return self.train_config.episode_num
    def _set_episode_num(self, val):
        if val is not None:
            self.train_config.episode_num = val
        if self.train_config.episode_num is None:
            raise RuntimeError(f'episode_num is None')
    episode_num = property(_get_episode_num, _set_episode_num)

    def _get_max_iter_in_episode(self): return self.train_config.max_iter_in_episode
    def _set_max_iter_in_episode(self, val):
        if val is not None:
            self.train_config.max_iter_in_episode = val
        if self.train_config.max_iter_in_episode is None:
            raise RuntimeError(f'max_iter_in_episode is None')
    max_iter_in_episode = property(_get_max_iter_in_episode, _set_max_iter_in_episode)

    def train(self, episode_num=None, max_iter_in_episode=None):
        self.episode_num = episode_num
        self.max_iter_in_episode = max_iter_in_episode
        for i_episode in range(episode_num):
            self.episode_process()

    def reset(self):
        state = self.env.reset()
        self.net.reset()
        del self.rewards[:]
        self.rewards = []
        return state

    def episode_process(self):
        state = self.reset()
        for i in range(self.max_iter_in_episode):
            action = self.net(state)
            state, reward, done, extra_info = self.env.step(action)
            if done:
                break

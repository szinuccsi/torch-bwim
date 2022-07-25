import torch
import torch.nn.functional as F

from torch_bwim.loss_functions.LossFunction import LossFunction


class ReinforcementLoss(LossFunction):

    class Config(LossFunction.Config):
        def __init__(self, gamma, epsilon=None):
            super().__init__()
            self.gamma = gamma
            self.epsilon = epsilon if epsilon is not None else 1e-10

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.config = config

    def __call__(self, saved_actions, env_rewards):
        cfg = self.config
        R = 0
        gamma = cfg.gamma
        policy_losses = []
        value_losses = []
        rewards = []
        for r in env_rewards[::-1]:
            R = r + gamma * R
            rewards.append(R)
        rewards.reverse()

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (torch.std(rewards) + torch.full_like(rewards, fill_value=cfg.epsilon))

        for saved_act, r in zip(saved_actions, rewards):
            new_policy_loss = self.policy_loss(saved_action=saved_act, reward=r)
            new_value_loss = self.value_loss(saved_action=saved_act, reward=r)
            policy_losses.append(new_policy_loss)
            value_losses.append(new_value_loss)
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def policy_loss(self, saved_action, reward):
        log_prob, value = saved_action
        corrigated_reward = reward - value.item()
        return -log_prob * corrigated_reward

    def value_loss(self, saved_action, reward):
        log_prob, value = saved_action
        return F.smooth_l1_loss(value, torch.tensor([reward]))

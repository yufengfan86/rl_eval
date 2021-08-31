"""
该文件定义actor-critic结构
"""
import torch
import torch.nn as nn

from models.mlp import MLP
from distributions.distributions import Gaussian


class AC(nn.Module):

    def __init__(self,
                 base_nn: str,
                 obs_shape: tuple,
                 hidden_state_shape: tuple,
                 hidden_feature_shape: tuple,
                 action_shape: tuple,
                 action_type: str
                 ):
        """
        :param base_nn:
        :param obs_shape:
        :param hidden_state_shape:
        :param hidden_feature_shape:
        :param action_shape:
        :param action_type:连续(continue)或离散(discrete)
        """
        super(AC, self).__init__()
        if base_nn == "mlp":
            self.base_nn = MLP(obs_shape, hidden_state_shape, hidden_feature_shape)
        else:
            raise NotImplemented

        if action_type == "continue":
            self.dist = Gaussian(hidden_feature_shape, action_shape)
        else:
            raise NotImplemented

    def act(self, obs, hidden_states, masks, deterministic=False):
        action_feature, h_n, value = self.base_nn(obs, hidden_states, masks)

        dist = self.dist(action_feature)

        if deterministic:  # 确定策略
            action = dist.mode()
        else:  # 不确定性策略
            action = dist.sample()

        log_probs = dist.log_probs(action)

        return value, action, h_n, log_probs

    def evaluate_action(self, obs, hidden_states, masks, actions):
        action_feature, h_n, value = self.base_nn(obs, hidden_states, masks)

        dist = self.dist(action_feature)

        log_probs = dist.log_probs(actions)

        dist_entropy = dist.entropy().mean()

        return value, log_probs, dist_entropy, h_n

    def get_value(self, obs, hidden_states, masks):
        _, _, value = self.base_nn(obs, hidden_states, masks)

        return value

    def forward(self):
        raise NotImplemented


if __name__ == '__main__':
    ac = AC("mlp", (3, 3), (2, 2), (3, 3), (1, ), "continue")

    obs = torch.randn((4, 3, 3))
    hidden_state = torch.randn((4, 2, 2))

    # masks = torch.ones(4, 1)

    # v, a, h, l = ac.act(obs, hidden_state, masks)

    # print(v, a, h, l)
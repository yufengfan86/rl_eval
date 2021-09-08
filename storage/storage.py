"""
该模块用来定义存储模块:
由于所有更新相关数据都存放在这里, 所以应该有相应的计算方法.
定义类按照ppo算法定义, 该类也可以用于其他算法.

每一个实例保存多个actor采集到的样本.

"""

import numpy as np

from utils.transform import func_n2t
from torch.utils.data import BatchSampler, SubsetRandomSampler


class Storage(object):

    def __init__(self, obs_shape, action_shape, hidden_state_shape, num_workers, n_steps):
        """
        :param obs_shape: should be a tuple.
        :param action_shape:
        :param hidden_state_shape:
        :param num_workers:
        :param n_steps:
        """
        self.obs_vec = np.zeros((num_workers, n_steps+1, *obs_shape))
        self.actions_vec = np.zeros((num_workers, n_steps, *action_shape))
        self.hidden_states_vec = np.zeros((num_workers, n_steps+1, *hidden_state_shape))
        self.rewards_vec = np.zeros((num_workers, n_steps, 1))

        # 这里是critic学习的value值
        self.values_vec = np.zeros((num_workers, n_steps+1, 1))

        # 这个log_prob可以通过.exp转换为prob.
        self.log_probs_vec = np.zeros((num_workers, n_steps, 1))

        # 使用reward计算得到的returns: r_t = r_{t+1} + ... + ...
        self.returns_vec = np.zeros((num_workers, n_steps, 1))

        # 用来记录该状态是否为结束状态, 0表示结束, 1表示正常
        self.masks_vec = np.ones((num_workers, n_steps+1, 1))

        self.worker_step_map = {i: 0 for i in range(num_workers)}

        self.max_step = n_steps
        self.num_workers = num_workers

    def push(self, worker, obs, action, hidden_state, reward, value, log_probs, mask):
        step_ = self.worker_step_map[worker]
        self.obs_vec[worker, step_+1] = obs
        self.actions_vec[worker, step_] = action
        self.hidden_states_vec[worker, step_+1] = hidden_state
        self.rewards_vec[worker, step_] = reward
        self.values_vec[worker, step_] = value
        self.log_probs_vec[worker, step_] = log_probs
        self.masks_vec[worker, step_+1] = mask

        self.worker_step_map[worker] = (self.worker_step_map[worker] + 1) % self.max_step

    def retrive_act_data(self, worker, step):
        obs = self.obs_vec[worker, step]
        hidden_state = self.hidden_states_vec[worker, step]
        mask = self.masks_vec[worker, step]

        return func_n2t((obs, hidden_state, mask))

    def after_update(self):
        self.obs_vec[:, 0] = self.obs_vec[:, -1]
        self.hidden_states_vec[:, 0] = self.hidden_states_vec[:, -1]
        self.masks_vec[:, 0] = self.masks_vec[:, -1]

    def cal_returns(self,
                    gamma: float,
                    gae_lambda: float,
                    t_values: np.ndarray
                    ):
        """
        参考gae
        :param gamma:
        :param gae_lambda:
        :param t_values: 该值为规定步数内的最后一个状态的value, 应该是所有workers的.
        :return:
        """
        gae = np.zeros((self.num_workers, 1))
        self.returns_vec[:, -1] = t_values

        for step_ in reversed(range(self.max_step)):
            delta = self.rewards_vec[:, step_] + \
                    gamma * self.values_vec[:, step_+1] * self.masks_vec[:, step_+1] - self.values_vec[:, step_]

            gae = delta + gamma * gae_lambda * gae * self.masks_vec[:, step_+1]

            self.returns_vec[:, step_] = gae + self.values_vec[:, step_]

    def sample_generator(self,
                         advantages_vec: np.ndarray,
                         mini_batch_size: int
                         ):

        samples_size = self.num_workers * self.max_step

        sampler = BatchSampler(
            SubsetRandomSampler(range(samples_size)),
            mini_batch_size,
            drop_last=True
        )  # tensor 作为索引可以直接应用在ndarray中.

        for indices in sampler:
            obs_batch = self.obs_vec[:, :-1].reshape(-1, *self.obs_vec.shape[2:])[indices]
            hidden_states_batch = self.hidden_states_vec[:, :-1].reshape(-1, *self.hidden_states_vec.shape[2:])[indices]
            actions_batch = self.actions_vec.reshape(-1, *self.actions_vec.shape[2:])[indices]

            values_batch = self.values_vec[:, :-1].reshape(-1, *self.values_vec.shape[2:])[indices]
            returns_batch = self.returns_vec.reshape(-1, *self.returns_vec.shape[2:])[indices]
            masks_batch = self.masks_vec[:, :-1].reshape(-1, *self.masks_vec.shape[2:])[indices]

            old_action_log_probs_batch = self.log_probs_vec.reshape(-1, *self.log_probs_vec.shape[2:])[indices]

            advantages_batch = advantages_vec.reshape(-1, *advantages_vec.shape[2:])[indices]

            yield obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, \
                  masks_batch, old_action_log_probs_batch, advantages_batch


if __name__ == '__main__':
    import torch
    from utils.transform import func_n2t
    obs_shape = (1, )
    action_shape = (1, )
    hidden_shape = (1, )
    num_workers = 5
    n_steps = 10

    storage = Storage(obs_shape, action_shape, hidden_shape, num_workers, n_steps)
    storage.obs_vec[0, 0] = 1
    storage.push(0, 1, 1, 1, 1, 1, 1, 1)
    storage.push(0, 1, 1, 1, 1, 1, 1, 1)
    storage.push(1, 1, 1, 1, 1, 1, 1, 1)

    storage.cal_returns(0.99, 0.8, np.ones((5, 1)))
    # print(storage.returns_vec)
    # print(storage.obs_vec[0])
    advantages = storage.returns_vec - storage.values_vec[:, :-1]
    # print(torch.from_numpy(advantages).shape)

    s = storage.sample_generator(advantages, 5)
    for ss in s:
        obs, *a = ss
        # print(obs.shape)
        print(torch.from_numpy(obs).shape)

    # storage.after_update()
    # print(storage.obs_vec[0])
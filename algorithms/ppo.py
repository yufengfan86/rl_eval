"""
ppo algorithm.
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + (gamma * lambda) * delta_{t+1} + .. + .. + (gamma * lambda)^{T-t+1} * delta_{T-1}

ratio_t = pi_{theta}(a_t|s_t) / pi_{old}(a_t|s_t)
L^{clip}(theta) = E_t [min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps)A_t]

L_t^{clip+VF+S) (theta) = E_t [L_t^{clip}(theta) - c1*L_t^{VF}(theta) + c2*S[pi(theta)(st)]

L_t^{VF} = (V_{theta} (st) - V_t^{target})**2
"""
import torch
import torch.optim as optim

from utils.transform import func_n2t, func_to


class PPO(object):

    def __init__(self,
                 actor_critic,
                 lr: float,
                 mini_batch_size: int,
                 clip_eps: float,
                 critic_coef: float,
                 entropy_coef: float,
                 update_epochs: int,
                 max_grad_norm: float,
                 eps = 1e-8
                 ):
        self.actor_critic = actor_critic
        self.mini_batch_size = mini_batch_size

        self.clip_eps = clip_eps
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, storage, gamma=0.99, gae_lambda=0.96):
        """
        storage中的所有数据为ndarray格式. 每个数据的shape: (num_works, steps, *data_shape).
        :param storage:
        :param gamma:
        :param gae_lambda:
        :return:
        """
        storage.cal_returns(gamma, gae_lambda, storage.values_vec[:, -1])

        advantage_vec = storage.returns_vec - storage.values_vec[:, :-1]
        # 标准化
        advantage_vec = (advantage_vec - advantage_vec.mean()) / (advantage_vec.std() + 1e-5)

        for epoch in range(self.update_epochs):
            sample_generator = storage.sample_generator(advantage_vec, self.mini_batch_size)

            for sample in sample_generator:
                obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, masks_batch, \
                old_action_log_probs_batch, advantages_batch = func_to(func_n2t(sample), device="0")

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_action(
                    obs_batch, hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1. - self.clip_eps, 1. + self.clip_eps)*advantages_batch

                clip_loss = -torch.min(surr1, surr2).mean()

                vf_loss = (returns_batch - values).pow(2).mean()

                loss = clip_loss + self.critic_coef*vf_loss - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

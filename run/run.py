"""
单进程
运行流程:
1. 创建环境, ac(agent), storage.
2. 采样 并 存入 storage
3. 更新
"""
import numpy as np

from environments.basic import func_generate_env
from utils.env_utils import func_env_infos
from storage.storage import Storage
from algorithms.ppo import PPO

from ac.ac import AC


def main(args):

    env = func_generate_env(args.env_name)

    obs_shape, action_shape, action_type = func_env_infos(env)

    hidden_state_shape = obs_shape

    hidden_feature_shape = tuple(map(lambda x: x*2, obs_shape))

    actor_critic = AC(args.base_nn, obs_shape, hidden_state_shape, hidden_feature_shape,
                      action_shape, action_type)

    alg = PPO(
        actor_critic,
        args.lr,
        args.mini_batch_size,
        args.clip_eps,
        args.critic_coef,
        args.entropy_coef,
        args.update_epochs,
        args.max_grad_norm
    )

    storage = Storage(obs_shape, action_shape, hidden_state_shape, 1, args.num_steps)

    storage.obs_vec[:, 0] = env.reset()

    for num_update_ in range(int(args.num_updates)):

        for step_ in range(args.num_steps):
            # env.render()
            value, action, h_n, log_probs = actor_critic.act(*storage.retrive_act_data(0, step_))
            next_obs, reward, done, infos = env.step(action.numpy()[0])

            if done:
                masks = np.zeros((1, 1))
            else:
                masks = np.ones((1, 1))

            assert infos['TimeLimit.truncated'] is False

            storage.push(0, next_obs, action.numpy(), h_n.detach().numpy(),
                         reward, value.detach().numpy(), log_probs.detach().numpy(), masks)

        value = actor_critic.get_value(*storage.retrive_act_data(0, args.num_steps))
        storage.values_vec[:, -1] = value.detach().numpy()
        alg.update(storage)

        print(num_update_, storage.rewards_vec.mean())

    env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")

    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--num_steps", type=int, default=10)

    parser.add_argument("--num_updates", type=int, default=10000)

    parser.add_argument("--base_nn", type=str, default="mlp")

    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--mini_batch_size", type=int, default=5)

    parser.add_argument("--clip_eps", type=float, default=0.2)

    parser.add_argument("--critic_coef", type=float, default=0.5)

    parser.add_argument("--entropy_coef", type=float, default=0.0001)

    parser.add_argument("--update_epochs", type=int, default=5)

    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    args = parser.parse_args()

    main(args)

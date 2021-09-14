"""
生成多个环境进行采样

1. 生成多个独立环境
2. 每个进程保持各自的环境
3. 进程接受不同的指令及数据, 并返回相应的数据.
4. 并行环境的使用方法, 应该与单个环境的使用方法相同或相似.
"""
import numpy as np
import multiprocessing as mp

from environments.basic import func_generate_env


class PEnv(object):
    # 应该接受一批pipe。
    # 数据流：每个pipe向另一端发送指令，并接受相应的返回值
    # reset：发送reset指令， 并获取每个env（eid）的初始化状态， 返回时， 应该已经是按照顺序打包好的数据
    # step： 发送step指令，及每个eid对应的action，所以， action需要按顺序发送， 返回时， 依旧是按顺序打包好的数据

    def __init__(self,
                 pipe_list,
                 ):
        self.pipe_list = pipe_list

    def reset(self):
        for pipe in self.pipe_list:  # 这里应该是异步的？还是串行的？如果是串行的，那么似乎没必要使用多进程。
            p = mp.Process(target=self._reset, args=(pipe, ))
            p.start()

        obe_list = []
        for pipe in self.pipe_list:
            obe_list.append(pipe.recv())

        ob_list = []
        for i in range(len(obe_list)):
            ob_list.append(obe_list[i][1])

        return np.array(ob_list)

    def _reset(self, pipe):
        pipe.send(["reset", None])

    def step(self,
             actions: np.ndarray
             ):
        # 这里的actions应该是已经按照eid排序好的。
        for eid, pipe in enumerate(self.pipe_list):  # 这里应该是异步的？还是串行的？如果是串行的，那么似乎没必要使用多进程。
            p = mp.Process(target=self._step, args=(pipe, actions[eid]))
            p.start()

        datae_list = []
        for pipe in self.pipe_list:
            datae_list.append(pipe.recv())

        data_list = []
        next_ob_list = []
        reward_list = []
        done_list = []
        info_list = []
        for i in range(len(datae_list)):
            next_ob_list.append(datae_list[i][1][0])
            reward_list.append(datae_list[i][1][1])
            done_list.append(datae_list[i][1][2])
            info_list.append(datae_list[i][1][3])

        return np.array(next_ob_list), np.array(reward_list), np.array(done_list), np.array(info_list)

    def _step(self,
              pipe,
              action
              ):
        pipe.send(["step", action])

    def __len__(self):
        return len(self.pipe_list)


def worker(pipe, remote_pipe, env, eid):

    def _step(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    remote_pipe.close()

    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                ob = env.reset()
                pipe.send((eid, ob))
            elif cmd == "step":
                next_ob, reward, done, info = _step(env, data)
                pipe.send((eid, [next_ob, reward, done, info]))
            elif cmd == "render":
                env.render()
            elif cmd == "close":
                pipe.close()
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("close")


def make_parallel_envs(env_name, num_workers):

    # 创建envs, eid为环境id
    envs = {eid: func_generate_env(env_name) for eid in range(num_workers)}

    # 为每一个环境创建一个pipe.
    pipes = [mp.Pipe() for _ in range(num_workers)]

    p_list = []

    pipe_list = []
    for ind in range(num_workers):
        p = mp.Process(target=worker, args=(pipes[ind][1], pipes[ind][0], envs[ind], ind))
        p.start()
        pipes[ind][1].close()
        p_list.append(p)
        pipe_list.append(pipes[ind][0])

    # for p in p_list:
    #     p.join()

    return pipe_list


if __name__ == '__main__':
    # 验证pipe是否两端都可以进行数据的接受与传送(可以)
    env = func_generate_env("CartPole-v0")

    envs = make_parallel_envs("CartPole-v0", 2)

    obs = envs.reset()

    while True:
        actions = []
        for i in range(len(envs)):
            actions.append(env.action_space.sample())

        actions = np.array(actions)

        obs, rewards, dones, infos = envs.step(actions)

        print(dones)










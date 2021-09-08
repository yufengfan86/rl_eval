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
    # pipe, remote_pipe = mp.Pipe()
    #
    env = func_generate_env("CartPole-v0")
    #
    # p = mp.Process(target=worker, args=(remote_pipe, pipe, env))
    #
    # p.start()
    #
    # remote_pipe.close()
    #
    # pipe.send(["reset", None])
    # obs = pipe.recv()
    # print(obs)
    #
    # while True:
    #     pipe.send(["render", None])
    #     data = env.action_space.sample()
    #     pipe.send(["step", data])
    #
    #     info = pipe.recv()
    #     print(info)

    # pipe.send(["close", None])
    pipe_list = make_parallel_envs("CartPole-v0", 2)

    for pipe in pipe_list:
        pipe.send(["reset", None])

    while True:
        data = env.action_space.sample()
        pipe_list[0].send(["step", data])
        print(pipe_list[0].recv())
        pipe_list[1].send(["step", data])
        print(pipe_list[1].recv())










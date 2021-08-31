"""
gym自带的环境
"""
import gym


def func_generate_env(env_name: str):

    env = gym.make(env_name)

    return env

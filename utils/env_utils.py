"""
定义一些用于获取环境相关信息的函数
"""


def func_env_infos(env:object):

    action_space = env.action_space

    if action_space.__class__.__name__ == "Discrete":
        action_shape = tuple((action_space.n, ))
        action_type = "discrete"
    elif action_space.__class__.__name__ == "Box":
        action_shape = action_space.shape
        action_type = "continue"
    elif action_space.__class__.__name__ == "MultiBinary":
        action_shape = action_space.shape
        action_type = "continue"
    else:
        raise NotImplementedError

    obs_shape = env.observation_space.shape

    return tuple(obs_shape), action_shape, action_type

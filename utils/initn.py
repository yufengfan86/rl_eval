"""
定义初始化网络参数的函数
"""


def func_init(module, weight_init, bias_init, gain):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)

    return module

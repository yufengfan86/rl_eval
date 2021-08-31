"""
这里定义一些常用的函数
作者: Mashun
"""
import torch

from builtins import tuple


def func_n2t(sample: tuple):
    sample = [torch.from_numpy(d).float() for d in sample]

    return sample


def func_t2n(sample: tuple):
    sample = [d.numpy() for d in sample]
    return sample


def func_to(sample,
            device: str):

    device = torch.device("cuda:{}".format(device)) if torch.cuda.is_available() else torch.device("cpu")

    sample = [d.to(device) for d in sample]

    return sample


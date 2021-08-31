# 结构

为了方便以后使用强化学习方法, 需要一个扩展性比较好的
强化学习框架.

## 设计思路

1. 各个文件及函数命名应保持与RL理论一致
2. 应该可以多进程并行(采样以及更新)
3. 可扩展性
    1. 可以替换不同的网络结构及强化学习算法
    2. 可以处理不同的环境


## 具体细节

### 文件结构

- models: 包含不同的网络结构
     - mlp.py: 全连接层网络  
     - cnn.py: 卷积神经网络

- algorithms: 包含不同的强化学习算法
     - ppo.py: ppo 算法
     - ddpg.py: ddgp 算法
     
- storage: 包含存储模块
     - storage.py
     
- utils: 包含一些常用的功能

- environments: 包含要使用的环境
     - basic.py: 基础的环境(gym)
     - others: 自定义环境
 
- ac: actor-critic结构 
     - ac.py: actor-critic结构 
     
- distributions: 定义方便使用的分布函数
     - distributions.py:  ...
     
- run: 定义运行方式
     - run.py: 正常运行
     - run_parallel.py: 多进程并行
     

      
      
# 环境研究

通过研究不同环境的不同状态(起始状态, 结束状态), 从而决定使用什么评价指标等相关问题.

####MountainCarContinuous-v0

    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Box(1)
        Num    Action                    Min            Max
        0      the power coef            -1.0           1.0
        Note: actual driving force is calculated by multipling the power coef by power (0.0015)
    Reward:
         Reward of 100 is awarded if the agent reached the flag (position = 0.45) on top of the mountain.
         Reward is decrease based on amount of energy consumed each step.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.45
         Episode length is greater than 200

    
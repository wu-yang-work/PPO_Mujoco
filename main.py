import os
import random

import gym
import numpy as np
import torch
from torch import optim
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


from network.net import Agent

seed = 2022

def env_fns():
    def make():
        env = gym.make("BreakoutNoFrameskip-v0")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        # 记录游戏每个epoch的奖励，运行长度，时间
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # 暂停30步
        env = NoopResetEnv(env, noop_max=30)
        # 跳过4帧
        env = MaxAndSkipEnv(env, skip=4)
        # 设置agent死掉一次就结束
        env = EpisodicLifeEnv(env)
        # 一直step到fire
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # 奖励裁剪
        env = ClipRewardEnv(env)
        # 观测reshape
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # 观测变成gray
        env = gym.wrappers.GrayScaleObservation(env)
        # 帧和并
        env = gym.wrappers.FrameStack(env, 1)
        return env
    return make

# os.add_dll_directory('C:\\Users\\14768\\.mujoco\mujoco210\\bin')
if __name__ == '__main__':

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    envs = gym.vector.SyncVectorEnv([env_fns()])
    # print(envs.observation_space.shape)

    print(envs.single_observation_space.shape)
    print(envs.single_action_space.shape)
    '''
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    learning_rate = 3e-4
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((2048, 1) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((2048, 1) + env.single_action_space.shape).to(device)
    logprobs = torch.zeros((2048, 1)).to(device)
    rewards = torch.zeros((2048, 1)).to(device)
    dones = torch.zeros((2048, 1)).to(device)
    values = torch.zeros((2048, 1)).to(device)
    '''
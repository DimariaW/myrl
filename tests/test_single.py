#%%
import myrl.utils as utils
import logging
utils.set_process_logger(file_path="./log/double_dqn_pr")

import gym

env = gym.make("LunarLander-v2")

obs_dim = env.observation_space.shape[0]
num_acts = env.action_space.n
#%%
from myrl.model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch


class DuelNet(Model):
    def __init__(self, obs_dim, num_acts):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, num_acts + 1)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        value_and_qvalue = self.fc3(h2)

        return value_and_qvalue[..., 0:1] + (value_and_qvalue[..., 1:] - torch.mean(value_and_qvalue[..., 1:], dim=-1, keepdim=True))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DuelNet(obs_dim=obs_dim, num_acts=num_acts).to(device)
#%%
from myrl.actor import Actor
from myrl.agent import DQNAgent

agent = DQNAgent(model, num_acts, device)
actor = Actor(env, agent, 100, get_full_episode=False, use_tensorboard=True, logdir="./log/actor/double_dqn_pr/")
#%%
from myrl.memory import MemoryReplay, PriorityMemoryReplay
import numpy as np
from myrl.algorithm import DQN

#tensor_receiver = MemoryReplay(100000, 512, device=device,  cached_in_device=True)
mr = PriorityMemoryReplay(100000, 512, device=device, cached_in_device=True)
dqn = DQN(model, mr, double_dqn=True)

while True:
    episode = actor.sample()
    for i, moment in enumerate(episode):
        if i < len(episode)-1:
            mr.cache(moment["observation"], moment["action"], episode[i+1]["observation"], moment["reward"], moment["done"])
        elif moment["done"] is True:
            mr.cache(moment["observation"], moment["action"], np.random.rand(*moment["observation"].shape),
                     moment["reward"], moment["done"])

    if len(mr) > mr.batch_size:
        logging.info(dqn.learn())



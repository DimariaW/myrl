#%%
import gym

env = gym.make("LunarLander-v2")

obs_dim = env.observation_space.shape[0]
num_acts = env.action_space.n

#%%
from rl.model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class Model(BaseModel):
    def __init__(self, obs_dim, num_acts):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, num_acts)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        return Q

model = Model(obs_dim, num_acts)
#%%

class Actor:
    def __init__(self, model, e_greed=0.1, act_dim=None):
        self.model = model
        self.e_greed = e_greed
        self.act_dim = act_dim

    def sample(self, obs):
        """Sample an action `for exploration` when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            act = torch.argmax(self.model(torch.from_numpy(obs).unsqueeze(0)), dim=1)[0].item()
        return act

actor = Actor(model, act_dim=num_acts)
#%%

from rl.memoryReplay import MemoryReplay
from rl.learner import DQN
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rpm = MemoryReplay(200000, device)
learner = DQN(copy.deepcopy(model).to(device), gamma=0.99, lr=5e-4, sync_time_delta=20)
#%%

total_steps = 1000000
obs = env.reset()
rs = 0
for step in range(total_steps):
    action = actor.sample(obs)
    next_obs, reward, done, _ = env.step(action)
    rpm.cache(obs, action, next_obs, reward, done)

    if (len(rpm) > 200) and (step % 5 == 0):
            # s,a,r,s',done
            s,a, s_, r, d = rpm.recall(64)
            train_loss = learner.learn(s, a, r, s_, d)
            if step % 20 == 0:
                weights = learner.model.get_weights()
                actor.model.set_weights(weights)

    rs += reward
    obs = next_obs
    if done:
        print(step, rs)
        rs = 0
        obs = env.reset()


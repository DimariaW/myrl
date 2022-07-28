import sys

import torch


import torch.nn as nn
import torch.nn.functional as F

import envs.env_wrapper as env_wrapper
import gym

from myrl.agent import IMPALAAgent, ACAgent
from myrl.model import Model
from myrl.actor import Actor
from myrl.utils import set_process_logger


class DuelNet(Model):
    def __init__(self, obs_dim, num_acts):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, num_acts + 1)

    def forward(self, obs):
        obs = list(obs.values())[0]
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        value_and_logits = self.fc3(h2)

        return value_and_logits[..., 0], value_and_logits[..., 1:]


if __name__ == "__main__":
    set_process_logger()
    env = env_wrapper.ScaleReward(gym.make("LunarLander-v2"), scale_factor=1/200)
    env = env_wrapper.DictObservation(env, key="state")
    env = env_wrapper.DictReward(env)
    obs_dim = env.observation_space.shape[0]
    num_acts = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelNet(obs_dim, num_acts).to(device)
    #agent = IMPALAAgent(model, device)
    agent = ACAgent(model, device)

    actor = Actor(env, agent, num_steps=0, num_episodes=2, get_full_episode=True)

    for _ in range(2):
        episode = actor.sample(model_id=0)

    infos = actor.episodes_infos

    for _ in range(2):
        actor.predict(model_id=1)

    infos = actor.episodes_infos
    print("...")


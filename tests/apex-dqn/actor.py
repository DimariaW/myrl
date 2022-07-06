#%%
import gym
from myrl.model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch
from myrl.agent import DQNAgent
from myrl.actor import Actor, ActorClient


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


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    num_acts = env.action_space.n

    device = torch.device("cpu")
    model = DuelNet(obs_dim=obs_dim, num_acts=num_acts).to(device)

    agent = DQNAgent(model, num_acts, device)
    actor = Actor(env, agent, 100, use_tensorboard=True, logdir="./log/actor/double_dqn_mp/")
    actor_client = ActorClient(actor, "172.18.237.67", 1234)
    actor_client.run()

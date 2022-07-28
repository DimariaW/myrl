
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from myrl.model import Model
from myrl.agent import ACAgent
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
        value_and_logits= self.fc3(h2)

        return value_and_logits[..., 0], value_and_logits[..., 1:]
#%%


if __name__ == "__main__":
    #envs = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    num_acts = env.action_space.n
    model = DuelNet(obs_dim, num_acts)

    worker = False
    if worker:
        device = torch.device("cpu")
        agent = ACAgent(model, device)
        actor = Actor(env, agent, 300)
        actor_client = ActorClient(actor, "localhost", 8989)
        while True:
            actor_client.run()









import gym
from myrl.model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch
from myrl.agent import IMPALAAgent
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


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    num_acts = env.action_space.n

    device = torch.device("cpu")
    model = DuelNet(obs_dim=obs_dim, num_acts=num_acts).to(device)

    agent = IMPALAAgent(model, device)
    actor = Actor(env, agent, steps=3000, get_full_episodes=True,
                  use_tensorboard=True, logdir="./log/impala/mp_batcher/")
    actor_client = ActorClient(actor, "127.0.1.1", 1234, role="evaluator")
    actor_client.run()


import gym
from myrl.model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch
from myrl.actor_server import MemoryReplayServer
from myrl.algorithm import PG, A2C, IMPALA
from myrl.memory_replay import TrajList
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
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        value_and_logits= self.fc3(h2)

        return value_and_logits[..., 0], value_and_logits[..., 1:]


if __name__ == "__main__":
    set_process_logger(file_path="./log/a2c.txt")
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    num_acts = env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DuelNet(obs_dim=obs_dim, num_acts=num_acts).to(device)
    mr = TrajList(device)
    learner = IMPALA(model, mr, lr=1e-3, ef=3e-5, vf=0.5)
    learner_server = MemoryReplayServer(learner, 1234, actor_num=10, sampler_num=10)
    learner_server.run_on_policy()

import torch
import torch.nn as nn
import logging

from myrl.model import Model


# orthogonal init
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Model(Model):
    def __init__(self, state_dim: int, num_act: int, use_orthogonal_init=True, use_tanh=True):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1 + num_act)
        if use_orthogonal_init:
            logging.info("use orthogonal init")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
        if use_tanh:
            logging.info("use tanh activate function")
            self.act_fn = nn.Tanh()
        else:
            self.act_fn = nn.ReLU()

    def forward(self, obs):
        feature = obs["feature"]
        h1 = self.act_fn(self.fc1(feature))
        h2 = self.act_fn(self.fc2(h1))
        output = self.fc3(h2)

        return output[..., 0], output[..., 1:]  # value and logit, value 的最后一维度需要squeeze


import torch
import torch.nn as nn
from myrl.algorithm import Algorithm
from myrl.memory_replay import MemoryReplay, PriorityMemoryReplay
from typing import Union
import copy
from myrl.model import Model
import logging


class DQN(Algorithm):
    def __init__(self, q_net: Model, memory_replay: Union[MemoryReplay, PriorityMemoryReplay],
                 double_dqn: bool = False, gamma: float = 0.99, lr: float = 1e-3,
                 sync_time_delta: int = 20, num_learns: int = 4):
        super().__init__()
        self.q_net = q_net
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.memory_replay = memory_replay

        self.gamma = gamma
        self.lr = lr
        self.sync_time_delta = sync_time_delta
        self.num_learns = num_learns
        self.method = "DoubleDQN" if double_dqn else "DQN"

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        if isinstance(self.memory_replay, PriorityMemoryReplay):
            self.loss_func = torch.nn.SmoothL1Loss(reduction="none")
        else:
            self.loss_func = torch.nn.SmoothL1Loss()

        self.learn_times = 0

    @torch.no_grad()
    def _q_target(self, next_state, reward, done):
        if self.method == "DoubleDQN":
            next_state_q = self.q_net(next_state)
            best_action = torch.argmax(next_state_q, dim=1, keepdim=True)
            next_q = self.q_target(next_state).gather(-1, best_action)
            return reward + (1.-done) * self.gamma * next_q
        else:
            next_q, _ = self.q_target(next_state).max(dim=1, keepdim=True)
            return reward + (1.-done) * self.gamma * next_q

    def _learn(self, state, action, next_state, reward, done, weight=None, index=None):
        td_target = self._q_target(next_state, reward, done)
        td_estimate = self.q_net(state).gather(-1, action)
        if weight is not None:
            loss, td_errors = self.update_q_net(td_estimate, td_target, self.loss_func, self.optimizer, weight=weight)
            self.memory_replay.update_priority(index, td_errors)
        else:
            loss = self.update_q_net(td_estimate, td_target, self.loss_func, self.optimizer)
        self.learn_times += 1
        if self.learn_times % self.sync_time_delta == 0:
            self.soft_update(self.q_target, self.q_net, tau=1.)
        return {"loss": loss, "qvalue": td_estimate.mean().item()}

    def learn(self):
        # 训练模式
        self.q_net.train()
        self.q_target.train()

        info = dict()
        for e in range(self.num_learns):
            info: dict = self._learn(*self.memory_replay.recall())
        return info

    def run(self):
        while True:
            if len(self.memory_replay) > self.memory_replay.batch_size:
                logging.info(self.learn())

    def get_weights(self):
        return self.q_net.get_weights()

    def set_weights(self, weights):
        return self.q_net.set_weights(weights)

from myrl.model import Model
from myrl.agent import Agent
import torch
import numpy as np


class DQNAgent(Agent):
    def __init__(self, q_net: Model, num_actions: int, device):
        super().__init__()
        self.q_net = q_net
        self.device = device
        self.epsilon = 0.9  # 贪婪程度
        self.num_actions = num_actions

    @torch.no_grad()
    def sample(self, state: np.ndarray):
        # EXPLORE
        if np.random.rand() < 1 - self.epsilon:
            action_idx = np.random.randint(self.num_actions, size=len(state))
        # EXPLOIT
        else:
            self.q_net.eval()
            state = torch.from_numpy(state).type(torch.float32).to(self.device)
            action_values = self.q_net(state)
            action_idx = torch.argmax(action_values, dim=1).cpu().numpy()
        return {"action": action_idx}

    @torch.no_grad()
    def predict(self, state: np.ndarray):

        self.q_net.eval()
        state = torch.from_numpy(state).type(torch.float32).to(self.device)
        action_values = self.q_net(state)
        action_idx = torch.argmax(action_values, dim=1).cpu().numpy()

        return {"action": action_idx}

    def set_weights(self, weights):
        self.q_net.set_weights(weights)

    def get_weights(self):
        return self.q_net.get_weights()

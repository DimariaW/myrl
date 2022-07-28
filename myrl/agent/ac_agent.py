
import torch
import numpy as np
from myrl.agent import Agent
from myrl.model import Model
from myrl.utils import to_tensor


class ACAgent(Agent):
    def __init__(self, model: Model, device):
        super().__init__()
        # force the model to device
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, state):  # batched np.ndarray
        self.model.eval()
        state = to_tensor(state, device=self.device)
        _, logits = self.model(state)
        action_idx = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        return {"action": action_idx}

    @torch.no_grad()
    def predict(self, state):
        self.model.eval()
        state = to_tensor(state, device=self.device)
        _, logits = self.model(state)
        action_idx = torch.argmax(logits, dim=-1).cpu().numpy()
        return {"action": action_idx}

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()


class IMPALAAgent(ACAgent):

    @torch.no_grad()
    def sample(self, state):
        self.model.eval()
        state = to_tensor(state, device=self.device)
        _, logits = self.model(state)
        action_idx = torch.distributions.Categorical(logits=logits).sample()
        log_prob = torch.log_softmax(logits, dim=-1)
        behavior_log_prob = torch.gather(log_prob, dim=-1, index=action_idx.unsqueeze(-1)).squeeze(-1)
        return {"action": action_idx.cpu().numpy(), "behavior_log_prob": behavior_log_prob.cpu().numpy()}


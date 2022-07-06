import torch
from myrl.model import Model
import logging


class Algorithm:
    def __init__(self):
        self.model = None
        self.memory_replay = None
        pass

    def learn(self):
        pass

    def run(self):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    @staticmethod
    def optimize(optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @staticmethod
    def gradient_clip_and_optimize(optimizer, loss, parameters, max_norm=40.0):
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        logging.debug(f"gradient norm is : {norm.item()}")
        optimizer.step()

    @staticmethod
    def update_q_net(q_estimate, q_target, loss_func, optimizer, weight=None):  # tensor shape: batch_size*1
        if weight is not None:
            loss = loss_func(q_estimate, q_target).squeeze(-1)
            #计算当前样本新的优先级
            td_errors = (loss.detach()).cpu().numpy()
            loss = torch.sum(loss * weight)
            Algorithm.optimize(optimizer, loss)
            return loss.item(), td_errors

        loss = loss_func(q_estimate, q_target)
        Algorithm.optimize(optimizer, loss)
        return loss.item()

    @staticmethod
    def soft_update(target_model: Model, model: Model, tau=0.005):
        model.sync_weights_to(target_model, 1 - tau)

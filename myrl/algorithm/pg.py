import torch
from myrl.model import Model
from myrl.algorithm import Algorithm
from myrl.memory_replay import TrajList, MultiProcessBatcher
import logging
from typing import Union

class PG(Algorithm):
    def __init__(self, model: Model, mr: TrajList, lr: float = 2e-3, gamma: float = 0.99):
        super().__init__()
        self.model = model
        self.memory_replay = mr

        self.lr = lr
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def learn(self):
        self.model.train()
        episodes = self.memory_replay.recall()
        obs = episodes["observations"]  # shape(B*T)
        action = episodes["actions"]

        value, logits = self.model(obs)

        action_log_probs = torch.log_softmax(logits, dim=-1)
        action_log_prob = torch.gather(action_log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)  # shape(B*T)

        loss = self.policy_gradient_loss(action_log_prob[:, :-1],
                                         episodes["rewards"][:, :-1],
                                         self.gamma,
                                         episodes["dones"][:, :-1],
                                         value.detach()[:, -1])
        self.optimize(self.optimizer, loss)

        self.memory_replay.empty()

        return {"loss": loss.item()}

    @staticmethod
    def policy_gradient_loss(action_log_prob: torch.Tensor,
                             reward: torch.Tensor,
                             gamma: float,
                             done: torch.Tensor,
                             bootstrap_value: torch.Tensor,):
        """
        calculate loss by vanilla policy_gradient algorithm.
        now the algorithm only support episodes case

        :param action_log_prob: shape(B, T)
        :param reward: shape(B, T)
        :param gamma: discount factor
        :param bootstrap_value: shape(B)
        :param done: shape(B, T)

        :return: loss: torch.Tensor scalar
        """

        cumulative_reward = []
        next_reward = bootstrap_value

        for i in range(action_log_prob.shape[1] - 1, -1, -1):
            curr_reward = reward[:, i]
            curr_done = done[:, i]
            cumulative_reward.insert(0, curr_reward + gamma * (1. - curr_done) * next_reward)

            next_reward = cumulative_reward[0]

        cumulative_reward = torch.stack(cumulative_reward, dim=-1)  # shape(B, T)

        return torch.mean(-action_log_prob * cumulative_reward)


class A2C(Algorithm):
    def __init__(self, model: Model, mr: Union[MultiProcessBatcher, TrajList],
                 lr: float = 2e-3, gamma: float = 0.99, lbd: float = 0.98, vf: float = 0.5, ef: float = 1e-3):
        super().__init__()
        self.model = model
        self.memory_replay = mr

        self.lr = lr
        self.gamma = gamma
        self.lbd = lbd
        self.vf = vf
        self.ef = ef

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        #self.critic_loss_fn = torch.nn.MSELoss()
        self.critic_loss_fn = torch.nn.SmoothL1Loss()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def learn(self):
        self.model.train()
        episodes = self.memory_replay.recall()

        obs = episodes["observations"]  # shape(B*T)
        action = episodes["actions"]
        reward = episodes["rewards"]
        done = episodes["dones"]

        value, logits = self.model(obs)

        entropy = torch.mean(torch.distributions.Categorical(logits=logits).entropy())

        action_log_probs = torch.log_softmax(logits, dim=-1)
        action_log_prob = torch.gather(action_log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)  # shape(B*T)

        value_nograd = value.detach()
        with torch.no_grad():
            adv, value_estimate = self.a2c_v1(value_nograd[:, :-1], reward[:, :-1], self.gamma, self.lbd, done[:, :-1], value_nograd[:, -1])

        actor_loss = torch.mean(-action_log_prob[:, :-1] * adv)
        critic_loss = self.critic_loss_fn(value[:, :-1], value_estimate)

        self.gradient_clip_and_optimize(self.optimizer, actor_loss + self.vf * critic_loss - self.ef * entropy, self.model.parameters(), max_norm=40.0)

        self.memory_replay.empty()

        return {"loss_actor": actor_loss.item(), "loss_critic": critic_loss.item(), "entropy": entropy.item()}

    @staticmethod
    def a2c_v1(value: torch.Tensor, reward: torch.Tensor,
               gamma: float, lbd: float, done: torch.Tensor, bootstrap_value: torch.Tensor):  # shape(B*T)

        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], bootstrap_value.unsqueeze(-1)], dim=-1) - value

        advantage = []
        next_adv = 0

        for i in range(value.shape[1]-1, -1, -1):
            curr_td_error = td_error[:, i]
            curr_done = done[:, i]

            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * next_adv)

            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=-1)

        return advantage, advantage+value

    @staticmethod
    def a2c_v2(value: torch.Tensor, reward: torch.Tensor,
               gamma: float, lbd: float, done: torch.Tensor, bootstrap_value: torch.Tensor):

        #n_step_advantage = []
        td_lbd_advantage = []

        #next_n_step_adv = 0
        next_td_lbd_adv = 0
        next_value = bootstrap_value

        for i in range(value.shape[1] - 1, -1, -1):
            curr_reward = reward[:, i]
            curr_value = value[:, i]
            curr_done = done[:, i]

            curr_td = curr_reward + (1. - curr_done) * gamma * next_value - curr_value
            #n_step_advantage.insert(0, curr_td + (1. - curr_done) * gamma * next_n_step_adv)
            td_lbd_advantage.insert(0, curr_td + (1. - curr_done) * gamma * lbd * next_td_lbd_adv)

            #next_n_step_adv = n_step_advantage[0]
            next_td_lbd_adv = td_lbd_advantage[0]

            next_value = curr_value

        #n_step_advantage = torch.stack(n_step_advantage, dim=-1)  # shape(B,T)
        td_lbd_advantage = torch.stack(td_lbd_advantage, dim=-1)  # shape(B, T)

        return td_lbd_advantage, td_lbd_advantage+value

    @staticmethod
    def a2c_v3(value: torch.Tensor, reward: torch.Tensor,
               gamma: float, lbd: float, done: torch.Tensor, bootstrap_value: torch.Tensor):

        td_lambda_value = []
        next_value = bootstrap_value
        next_td_lambda_value = bootstrap_value

        for i in range(value.shape[1]-1, -1, -1):
            curr_reward = reward[:, i]
            curr_done = done[:, i]

            td_lambda_value.insert(0, (curr_reward + gamma * (1.-curr_done) *
                                   ((1-lbd) * next_value + lbd * next_td_lambda_value)))

            next_value = value[:, i]
            next_td_lambda_value = td_lambda_value[0]

        td_lambda_value = torch.stack(td_lambda_value, dim=-1)

        return td_lambda_value - value, td_lambda_value


class IMPALA(A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self):
        self.model.train()
        episodes = self.memory_replay.recall()

        obs = episodes["observations"]  # shape(B*T)
        behavior_log_prob = episodes['behavior_log_probs']
        action = episodes["actions"]
        reward = episodes["rewards"]
        done = episodes["dones"]

        value, action_logit = self.model(obs)  # shape: B*T, B*T*act_dim

        action_log_prob = torch.log_softmax(action_logit, dim=-1)
        action_log_prob = action_log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1) # B*T

        log_rho = action_log_prob.detach() - behavior_log_prob
        rho = torch.exp(log_rho)

        logging.debug(f" rho is {torch.mean(rho)}")

        clipped_rho = torch.clamp(rho, 0, clip_rho_threshold := 1)  # rho shape: B*T
        cs = torch.clamp(rho, 0, clip_c_threshold := 1)  # c shape: B*T

        value_nograd = value.detach()

        vtrace_adv, vtrace_value = self.vtrace(value_nograd[:, :-1], reward[:, :-1], done[:, :-1],
                                               gamma=self.gamma, lbd=self.lbd, rho=clipped_rho[:, :-1], c=cs[:, :-1],
                                               bootstrap_value=value_nograd[:, -1])

        logging.debug(f" adv is {torch.mean(vtrace_adv)}")
        logging.debug(f" value is {torch.mean(vtrace_value)}")

        actor_loss = torch.mean(-action_log_prob[:, :-1] * clipped_rho[:, :-1] * vtrace_adv)

        critic_loss = self.critic_loss_fn(value[:, :-1], vtrace_value)

        entropy = torch.mean(torch.distributions.Categorical(logits=action_logit).entropy())

        loss = actor_loss + self.vf * critic_loss - self.ef * entropy

        self.gradient_clip_and_optimize(self.optimizer, loss, self.model.parameters(), 40.0)

        return {f"actor loss": actor_loss.item(), "critic loss": critic_loss.item(), "entropy": entropy.item()}

    @staticmethod
    @torch.no_grad()
    def vtrace(value, reward, done, gamma, lbd, bootstrap_value, rho, c):

        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], bootstrap_value.unsqueeze(-1)],
                                                               dim=-1) - value
        td_error = rho * td_error

        advantage = []
        next_adv = 0

        for i in range(value.shape[1] - 1, -1, -1):
            curr_td_error = td_error[:, i]
            curr_done = done[:, i]
            curr_c = c[:, i]
            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * curr_c * next_adv)

            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=-1)
        vtrace_value = advantage + value

        advantage = reward + gamma * (1. - done) * torch.concat([vtrace_value[:, 1:], bootstrap_value.unsqueeze(-1)],
                                                                 dim=-1) - value
        return advantage, vtrace_value

    def run(self):
        #is_started = False
        while True:
         #   if len(self.memory_replay) > 2 * self.memory_replay.batch_size:
          #      if not is_started:
           #         self.memory_replay.start()
            #        is_started = True
            logging.info(self.learn())


"""
class ACLearner:
    def __init__(self, model: Model, traj_replay: TrajReplay, lr=1e-2, gamma=0.99, lbd=1.):
        self.model = model
        self.traj_replay = traj_replay
        self.lr = lr
        self.gamma = gamma
        self.lbd = lbd
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [20000, 40000])

        self.value_loss_fn = torch.nn.MSELoss(reduction="none")
        self.global_train_steps = 0

    def learn(self):
        while True:
            self.model.train()
            batch = self.traj_replay.recall()

            values, action_logits = self.model(batch["observations"])  # shape: B*T, B*T*act_dim
            behavior_log_probs = batch['behavior_log_probs']

            action_log_probs = F.log_softmax(action_logits, dim=-1)
            action_log_probs = action_log_probs.gather(-1,
                                                       batch["actions"].unsqueeze(-1)).squeeze(-1)

            log_rhos = action_log_probs.detach() - behavior_log_probs
            rhos = torch.exp(log_rhos)
            clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold := 1)  # rho shape: B*T
            cs = torch.clamp(rhos, 0, clip_c_threshold := 1)  # c shape: B*T

            values_nograd = values.detach()

            n_step_vtrace_value, td_lbd_vtrace_value = self.vtrace(values_nograd, batch["rewards"], batch["dones"],
                                                                   batch["masks"], gamma=self.gamma, lbd=self.lbd,
                                                                   rho=clipped_rhos, c=cs)
            n_step_vtrace_q_value = batch["rewards"] +\
                                self.gamma * torch.concat([n_step_vtrace_value[:, 1:], n_step_vtrace_value[:, -1:]], dim=-1)

            td_lbd_vtrace_q_value = batch["rewards"] +\
                                self.gamma * torch.concat([td_lbd_vtrace_value[:, 1:], td_lbd_vtrace_value[:, -1:]], dim=-1)

            critic_mask = batch["bootstrap_masks"]
            actor_mask = batch["bootstrap_masks"]

            actor_loss = torch.sum(-action_log_probs * clipped_rhos * (n_step_vtrace_q_value-values_nograd)
                                   * (1. - actor_mask)) / torch.sum(1. - actor_mask)

            critic_loss = torch.sum(self.value_loss_fn(values, n_step_vtrace_value)
                                    * (1. - critic_mask)) / torch.sum(1. - critic_mask)

            entropy_loss = torch.sum(-1e-3 *
                                     torch.distributions.Categorical(logits=action_logits).entropy() *
                                     (1. - critic_mask)) / torch.sum(1. - critic_mask)

            loss = actor_loss + entropy_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_schedule.step()

            self.global_train_steps += 1
            print(self.global_train_steps, loss.item())

    @staticmethod
    def vtrace(values, rewards, dones, padding_masks, gamma, lbd, rho, c):

        n_step_advantage = []
        td_lbd_advantage = []

        next_n_step_adv = 0
        next_td_lbd_adv = 0
        next_value = 0

        for i in range(values.shape[1] - 1, -1, -1):
            curr_reward = rewards[:, i]
            curr_value = values[:, i]
            curr_done = dones[:, i]
            curr_rho = rho[:, i]
            curr_c = c[:, i]
            try:
                curr_mask = padding_masks[:, i + 1]
            except IndexError:
                curr_mask = 1.

            curr_td = curr_reward + (1. - curr_done) * gamma * (1. - curr_mask) * next_value - curr_value
            curr_td = curr_td * curr_rho

            n_step_advantage.insert(0, curr_td * (1. - curr_mask) + gamma * curr_c * next_n_step_adv)
            td_lbd_advantage.insert(0, curr_td * (1. - curr_mask) + gamma * lbd * curr_c * next_td_lbd_adv)

            next_n_step_adv = (1. - curr_mask) * n_step_advantage[0]
            next_td_lbd_adv = (1. - curr_mask) * td_lbd_advantage[0]

            next_value = curr_value

        n_step_advantage = torch.stack(n_step_advantage, dim=-1)  # shape(B,T)
        td_lbd_advantage = torch.stack(td_lbd_advantage, dim=-1)  # shape(B, T)

        return n_step_advantage + values, td_lbd_advantage + values

    def pg_learn(self):
        self.model.train()
        batch = self.traj_replay.recall()
        value, action_logits = self.model(batch["observations"])
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = action_log_probs.gather(-1, batch["actions"].type(torch.int64).unsqueeze(-1)).squeeze(-1)

        loss = self.policy_gradient_loss(action_log_probs, value.detach(),
                                         batch["rewards"], batch["masks"],
                                         batch["dones"], batch["tail_masks"], self.gamma)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @staticmethod
    def policy_gradient_loss(action_log_probs: torch.Tensor,
                             value: torch.Tensor,
                             rewards: torch.Tensor,
                             padding_masks: torch.Tensor,
                             dones: torch.Tensor,
                             tail_masks: torch.Tensor,
                             gamma: float = 0.99):
        """
"""
        calculate loss by vanilla policy_gradient algorithm.
        now the algorithm only support episodes case

        :param action_log_probs: shape(B, T)
        :param value: shape(B,T)
        :param rewards: shape(B, T)
        :param padding_masks: shape(B, T), padding is 1
        :param dones: shape(B,T)
        :param tail_masks: shape(B,T)
        :param gamma: discount factor

        :return: loss: torch.Tensor scalar
"""
"""
        cumulative_rewards = []
        next_value = 0

        for i in range(action_log_probs.shape[1] - 1, -1, -1):
            curr_reward = rewards[:, i]
            try:
                curr_mask = padding_masks[:, i + 1]
            except IndexError:
                curr_mask = 1.
            cumulative_rewards.insert(0, curr_reward + (1. - dones[:, i]) * gamma * (1. - curr_mask) * next_value)
            next_value = (1. - curr_mask)*cumulative_rewards[0] + curr_mask*value[:, i]

        cumulative_rewards = torch.stack(cumulative_rewards, dim=-1)  # shape(B,T)

        return torch.sum(-action_log_probs * cumulative_rewards * (1. - tail_masks)) / torch.sum(1. - tail_masks)

    def a2c_learn(self):
        self.model.train()
        batch = self.traj_replay.recall()
        value, action_logits = self.model(batch["observations"])
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = action_log_probs.gather(-1, batch["actions"].type(torch.int64).unsqueeze(-1)).squeeze(-1)

        loss = self.advantage_actor_critic_loss(action_log_probs, value,
                                                batch["rewards"], batch["masks"],
                                                batch["dones"], batch["tail_masks"],
                                                self.gamma, self.lbd,
                                                self.value_loss_fn,
                                                action_logits=action_logits)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @staticmethod
    def advantage_actor_critic_loss(action_log_probs: torch.Tensor,
                                    value: torch.Tensor,
                                    rewards: torch.Tensor,
                                    padding_masks: torch.Tensor,
                                    dones: torch.Tensor,
                                    tail_masks: torch.Tensor,
                                    gamma: float = 0.99,
                                    lbd: float = 0.98,
                                    critic_loss_func: Callable = None,
                                    entropy: float = 1e-3,
                                    action_logits: Optional[torch.Tensor] = None):
        value_nograd = value.detach()

        n_step_advantange = []
        td_lbd_advantange = []

        next_n_step_adv = 0
        next_td_lbd_adv = 0
        next_value = 0

        for i in range(action_log_probs.shape[1] - 1, -1, -1):
            curr_reward = rewards[:, i]
            curr_value = value_nograd[:, i]
            curr_done = dones[:, i]
            try:
                curr_mask = padding_masks[:, i + 1]
            except IndexError:
                curr_mask = 1.

            curr_td = curr_reward + (1. - curr_done) * gamma * (1. - curr_mask) * next_value - curr_value
            n_step_advantange.insert(0, curr_td + gamma*next_n_step_adv)
            td_lbd_advantange.insert(0, curr_td + gamma*lbd*next_td_lbd_adv)

            next_n_step_adv = (1. - curr_mask)*n_step_advantange[0]
            next_td_lbd_adv = (1. - curr_mask)*td_lbd_advantange[0]

            next_value = curr_value

        n_step_advantange = torch.stack(n_step_advantange, dim=-1)  # shape(B,T)
        td_lbd_advantange = torch.stack(td_lbd_advantange, dim=-1)  # shape(B, T)

        actor_loss = torch.sum(-action_log_probs*td_lbd_advantange * (1. - tail_masks)) / torch.sum(1. - tail_masks)
        critic_loss = torch.sum(critic_loss_func(value, n_step_advantange+value_nograd)
                                * (1. - tail_masks)) / torch.sum(1. - tail_masks)

        entropy_loss = torch.sum(-entropy *
                                 torch.distributions.Categorical(logits=action_logits).entropy() *
                                 (1. - tail_masks)) / torch.sum(1. - tail_masks)

        return actor_loss + entropy_loss + critic_loss

"""


if __name__ == "__main__":
    import random
    import time
    value = torch.randn((640, 1280))
    reward = torch.randn((640, 1280))
    gamma = random.random()
    lbd = random.random()
    done = torch.randint(low=0, high=2, size=(640, 1280))
    bootstrap_value = torch.randn(640)

    beg1 = time.time()
    a1, v1 = A2C.a2c_v1(value, reward, gamma, lbd, done, bootstrap_value)
    beg2 = time.time()
    a2, v2 = A2C.a2c_v2(value, reward, gamma, lbd, done, bootstrap_value)
    beg3 = time.time()
    a3, v3 = A2C.a2c_v3(value, reward, gamma, lbd, done, bootstrap_value)
    beg4 = time.time()
    print(torch.sum(a1-a2), torch.sum(a1-a3), torch.sum(a2-a3))
    print(torch.sum(v1 - v2), torch.sum(v1 - v3), torch.sum(v2 - v3))
    print(f"1 :{beg2-beg1}, 2:{beg3 - beg2}, 3:{beg4-beg3}")

from myrl.agent import Agent
import numpy as np
import myrl.connection as connection
from tensorboardX import SummaryWriter
from typing import Literal

__all__ = ["Actor", "ActorClient"]


class Actor:
    def __init__(self, env, agent: Agent,
                 steps: int = 50,
                 get_full_episodes: bool = False,
                 use_tensorboard=False,
                 logdir=None):
        """
        :param env:
        :param agent:
        :param steps: 采用当前模型采样steps步
        :param get_full_episodes: 若采样steps步之后，继续采样一定步数直到当前episode结束
        """
        self.env = env
        self.agent = agent

        self.steps = steps
        self.get_full_episodes = get_full_episodes

        self.global_episode = 0

        self.use_tensorboard = use_tensorboard

        if self.use_tensorboard:
            #self.actor_id = random.randint(0, 10000)
            #print(f"actor id is {self.actor_id}")
            self.sw = SummaryWriter(logdir=logdir)

        #self.hidden = None
        self.obs: np.array = self.env.reset()
        self.done = False
        self.current_episode_step = 0
        self.current_episode_total_reward = 0

    def sample(self):
        # episode generation
        episode = []
        step = 0

        while step < self.steps or (not self.done and self.get_full_episodes):

            moment = dict()
            moment['observation'] = self.obs.astype(np.float32)

            action_info = self.agent.sample(self.obs[np.newaxis, :])
            for key, value in action_info.items():
                moment[key] = value[0]


            #moment['value'] = value.item()

            #action_mask = torch.zeros((1, self.env.action_space.n))
            #action_logits = action_logits - action_mask

            #behavior_log_probs = F.log_softmax(action_logits, dim=-1)
            #action = Categorical(logits=behavior_log_probs).sample().item()

            #moment['action'] = action
            #moment['behavior_log_prob'] = behavior_log_probs[0, action].item()

            #moment['action_mask'] = action_mask.squeeze(0).numpy()
            #moment['action'] = action

            self.obs, reward, self.done, info = self.env.step(moment['action'])
            step += 1

            moment["reward"] = reward
            moment["done"] = self.done

            episode.append(moment)

            self.current_episode_step += 1
            self.current_episode_total_reward += reward

            if self.done:
                #self.hidden = None
                self.obs: np.ndarray = self.env.reset()
                self.done = False

                self.global_episode += 1
                print(f"global_episode is : {self.global_episode}, reward is : {self.current_episode_total_reward}")
                if self.use_tensorboard:
                    self.sw.add_scalar(tag=f"reward", scalar_value=self.current_episode_total_reward,
                                       global_step=self.global_episode)

                self.current_episode_step = 0
                self.current_episode_total_reward = 0

                if step >= self.steps:
                    break

        return episode

    def predict(self):
        step = 0
        episodes_reward = []
        while step < self.steps or (not self.done and self.get_full_episodes):

            action_info = self.agent.predict(self.obs[np.newaxis, :])

            self.obs, reward, self.done, info = self.env.step(action_info['action'][0])

            step += 1

            self.current_episode_step += 1
            self.current_episode_total_reward += reward

            if self.done:
                # self.hidden = None
                self.obs: np.ndarray = self.env.reset()
                self.done = False

                self.global_episode += 1

                print(f"global_episode is : {self.global_episode}, "
                      f"reward is : {self.current_episode_total_reward},"
                      f"episodes length is {self.current_episode_step}")

                episodes_reward.append(self.current_episode_total_reward)

                if self.use_tensorboard:
                    self.sw.add_scalar(tag=f"reward", scalar_value=self.current_episode_total_reward,
                                       global_step=self.global_episode)

                self.current_episode_step = 0
                self.current_episode_total_reward = 0

                if step >= self.steps:
                    break

        return np.mean(episodes_reward)


class ActorClient:
    def __init__(self, actor: Actor, host, port, role: Literal["sampler", "evaluator"] = "sampler"):
        self.actor = actor

        conn = connection.connect_socket_connection(host, port)
        if conn is not None:
            print(f"successfully connect to host: {host}, port: {port}")
        else:
            raise ConnectionError("fail to connect to host")
        self.conn = conn
        self.host = host
        self.port = port

        self.role = role

    def run(self):
        while True:
            cmd, weights = connection.send_recv(self.conn, ("model", None))
            # print(cmd)
            self.actor.agent.set_weights(weights)

            if self.role == "sampler":
                episode = self.actor.sample()
                cmd, info = connection.send_recv(self.conn, ("episodes", episode))
                print(cmd, info)
            elif self.role == "evaluator":
                outcome = self.actor.predict()
                print(outcome)


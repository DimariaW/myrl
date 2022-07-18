from myrl.agent import Agent
import numpy as np
import myrl.connection as connection
from tensorboardX import SummaryWriter
from typing import Literal, Callable
import logging
import multiprocessing as mp
from myrl.utils import batchify, set_process_logger
from collections import deque


__all__ = ["Actor", "ActorClient"]


class Actor:
    def __init__(self, env, agent: Agent,
                 steps: int = 50,
                 get_full_episodes: bool = False,
                 use_tensorboard=False,
                 logdir=None):
        """
        :param env:   gym api
        :param agent: 接受batched np.array 作为输入
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
        self.obs = self.env.reset()
        self.done = False
        self.current_episode_step = 0
        self.current_episode_total_reward = 0

        self.episode_rewards = []
        self.episode_steps = []

    def sample(self):
        # episode generation
        episode = []
        step = 0

        while step < self.steps or (not self.done and self.get_full_episodes):

            moment = dict()
            moment['observation'] = self.obs

            action_info = self.agent.sample(batchify([self.obs], unsqueeze=0))

            for key, value in action_info.items():
                moment[key] = value[0]

            #moment['value'] = value.item()

            #action_mask = torch.zeros((1, self.envs.action_space.n))
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
                self.obs = self.env.reset()
                self.done = False

                self.global_episode += 1
                logging.info(f"global_episode is : {self.global_episode}, "
                             f"reward is : {self.current_episode_total_reward},"
                             f"steps is : {self.current_episode_step},"
                             f"info is {info}")

                if self.use_tensorboard:
                    self.sw.add_scalar(tag=f"reward", scalar_value=self.current_episode_total_reward,
                                       global_step=self.global_episode)

                self.episode_steps.append(self.current_episode_step)
                self.episode_rewards.append(self.current_episode_total_reward)

                self.current_episode_step = 0
                self.current_episode_total_reward = 0

                if step >= self.steps:
                    break

        return episode

    def predict(self):
        step = 0
        episodes_reward = []
        while step < self.steps or (not self.done and self.get_full_episodes):

            action_info = self.agent.predict(batchify([self.obs], unsqueeze=0))

            self.obs, reward, self.done, info = self.env.step(action_info['action'][0])

            step += 1

            self.current_episode_step += 1
            self.current_episode_total_reward += reward

            if self.done:
                # self.hidden = None
                self.obs = self.env.reset()
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

        return episodes_reward


class ActorClient:
    def __init__(self,
                 actor_id: int,
                 actor: Actor,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 role: Literal["sampler", "evaluator"] = "sampler",
                 ):
        self.actor_id = actor_id
        self.actor = actor
        self.queue_gather2actor = queue_gather2actor
        self.queue_actor2gather = queue_actor2gather
        self.role = role

    def run(self):
        send_metric_interval = 10
        send_num = 0
        while True:
            self.queue_actor2gather.put((self.actor_id, "model", None))
            weights = self.queue_gather2actor.get()
            # print(cmd)
            self.actor.agent.set_weights(weights)

            if self.role == "sampler":
                episode = self.actor.sample()
                self.queue_actor2gather.put((self.actor_id, "episode", episode))
                logging.debug(self.queue_gather2actor.get())

                send_num += 1
                if send_num % send_metric_interval == 0:
                    if len(self.actor.episode_rewards) > 0:
                        self.queue_actor2gather.put((self.actor_id, "sample_reward", self.actor.episode_rewards))
                        logging.debug(self.queue_gather2actor.get())
                        self.actor.episode_rewards = []
                    send_num = 0

                #logging.info(self.queue_gather2actor.get())


class Gather:
    def __init__(self, gather_id: int, server_conn, num_sample_actors: int, num_predict_actors: int, func: Callable):
        super().__init__()
        self.gather_id = gather_id
        self.server_conn = server_conn

        self.cached_model_weights = None
        self.sent_cached_model_weights_times = 0
        self.max_sent_cached_model_weights_times = num_sample_actors*2

        self.cached_results = []
        self.max_cached_results_length = num_sample_actors * 4

        self.cached_episodes = []
        self.max_cached_episodes_length = 1 + num_sample_actors // 4

        self.queue_gather2actors = []
        self.queue_actor2gather = mp.Queue(maxsize=4 * (num_sample_actors + num_predict_actors))

        for i in range(num_sample_actors):
            self.queue_gather2actors.append(mp.Queue(maxsize=1))
            mp.Process(target=func, args=(i, self.queue_gather2actors[i], self.queue_actor2gather),
                       name=f"gather_{gather_id}_actor_{i}", daemon=True).start()

    def run(self):
        while True:

            actor_index, command, data = self.queue_actor2gather.get()
            actor_index: int

            if command == "model":
                if self.cached_model_weights is None or \
                        self.sent_cached_model_weights_times > self.max_sent_cached_model_weights_times:

                    self.cached_model_weights = connection.send_recv(self.server_conn, (command, data))
                    self.sent_cached_model_weights_times = 0
                self.queue_gather2actors[actor_index].put(self.cached_model_weights)
                self.sent_cached_model_weights_times += 1

            elif command == "episode":
                self.cached_episodes.append(data)
                if len(self.cached_episodes) > self.max_cached_episodes_length:
                    logging.debug(connection.send_recv(self.server_conn, (command, self.cached_episodes)))
                    self.cached_episodes = []
                self.queue_gather2actors[actor_index].put((command, "successfully receive episodes"))

            elif command == "sample_reward":
                self.cached_results.extend(data)
                if len(self.cached_results) > self.max_cached_results_length:
                    logging.debug(connection.send_recv(self.server_conn, (command, self.cached_results)))
                    self.cached_results = []
                self.queue_gather2actors[actor_index].put((command, "successfully receive sample reward"))


def _open_per_gather(gather_id, host, port, num_sample_actors, num_predict_actors, func):
    set_process_logger()  # windows need
    server_conn = connection.connect_socket_connection(host, port)
    logging.info(f"successfully connect to {host}:{port}! the gather {gather_id} is started!")
    Gather(gather_id, server_conn, num_sample_actors, num_predict_actors, func).run()


def open_gather(host: str, port: int, num_gathers: int,
                num_sample_actors_per_gather: int, num_predict_actors_per_gather: int = 0, func: Callable = None):
    processes = []
    for i in range(num_gathers):
        p = mp.Process(name=f"gather_{i}", target=_open_per_gather,
                       args=(i, host, port, num_sample_actors_per_gather, num_predict_actors_per_gather, func),
                       daemon=False
                       )
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.close()


def create_actor(actor_index: int, queue_gather2actor, queue_actor2gather):
    pass


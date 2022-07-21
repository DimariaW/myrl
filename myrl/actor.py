from myrl.agent import Agent
import myrl.connection as connection
from tensorboardX import SummaryWriter
from typing import Callable
import logging
import multiprocessing as mp
from myrl.utils import batchify, set_process_logger


__all__ = ["Actor", "ActorClient", "open_gather"]


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

                #self.episode_steps.append(self.current_episode_step)
                self.episode_rewards.append(self.current_episode_total_reward)

                self.current_episode_step = 0
                self.current_episode_total_reward = 0

                if step >= self.steps:
                    break

        return episode

    def predict(self):
        step = 0
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

                logging.info(f"global_episode is : {self.global_episode}, "
                             f"reward is : {self.current_episode_total_reward},"
                             f"episodes length is {self.current_episode_step},"
                             f"info is {info}"
                             )

                self.episode_rewards.append(self.current_episode_total_reward)
                #self.episode_steps.append(self.current_episode_step)

                if self.use_tensorboard:
                    self.sw.add_scalar(tag=f"reward", scalar_value=self.current_episode_total_reward,
                                       global_step=self.global_episode)

                self.current_episode_step = 0
                self.current_episode_total_reward = 0

                if step >= self.steps:
                    break
        return


class ActorClient:
    def __init__(self,
                 actor_id: int,
                 actor: Actor,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 role: str = "sampler",  # Literal["sampler", "evaluator"]
                 ):
        self.actor_id = actor_id
        self.actor = actor
        self.queue_gather2actor = queue_gather2actor
        self.queue_actor2gather = queue_actor2gather
        self.role = role

    def run(self):

        send_rewards_length = 5
        while True:
            self.queue_actor2gather.put((self.actor_id, "model", None))
            weights = self.queue_gather2actor.get()
            logging.debug(("model", "successfully request model weights"))
            self.actor.agent.set_weights(weights)

            if self.role == "sampler":
                episode = self.actor.sample()
                self.queue_actor2gather.put((self.actor_id, "episode", episode))
                logging.debug(self.queue_gather2actor.get())

                if len(self.actor.episode_rewards) > send_rewards_length:
                    self.queue_actor2gather.put((self.actor_id, "sample_reward", self.actor.episode_rewards))
                    logging.debug(self.queue_gather2actor.get())
                    self.actor.episode_rewards = []

            elif self.role == "evaluator":
                self.actor.predict()

                if len(self.actor.episode_rewards) > send_rewards_length:
                    self.queue_actor2gather.put((self.actor_id, "eval_reward", self.actor.episode_rewards))
                    logging.debug(self.queue_gather2actor.get())
                    self.actor.episode_rewards = []


class Gather:
    def __init__(self,
                 gather_id: int,
                 server_conn,
                 num_sample_actors: int,
                 num_predict_actors: int,
                 func: Callable):
        super().__init__()
        self.gather_id = gather_id
        self.server_conn = server_conn

        self.cached_model_weights = None
        self.sent_cached_model_weights_times = 0
        self.max_sent_cached_model_weights_times = num_sample_actors * 16

        self.cached_sample_reward = []
        self.max_cached_sample_reward_length = num_sample_actors * 5

        self.cached_eval_reward = []
        self.max_cached_eval_reward_length = num_predict_actors * 5

        self.cached_episodes = []
        self.max_cached_episodes_length = 1 + num_sample_actors//2

        self.queue_gather2actors = []
        self.queue_actor2gather = mp.Queue(maxsize=2 * (num_sample_actors + num_predict_actors))

        for i in range(num_sample_actors + num_predict_actors):
            self.queue_gather2actors.append(mp.Queue(maxsize=1))
            mp.Process(target=func, args=(i, self.queue_gather2actors[i], self.queue_actor2gather),
                       name=f"gather_{gather_id}_actor_{i}", daemon=True).start()

    def run(self):
        while True:

            actor_index, command, data = self.queue_actor2gather.get()
            actor_index: int
            logging.debug(f"actor_index : {actor_index}, command : {command}")

            if command == "model":
                if self.cached_model_weights is None or \
                        self.sent_cached_model_weights_times > self.max_sent_cached_model_weights_times:

                    cmd, self.cached_model_weights = connection.send_recv(self.server_conn, (command, data))
                    logging.debug(f"successfully request {cmd}")
                    self.sent_cached_model_weights_times = 0
                self.queue_gather2actors[actor_index].put(self.cached_model_weights)
                self.sent_cached_model_weights_times += 1

            elif command == "episode":
                self.cache_and_send_data(self.cached_episodes, [data], self.max_cached_episodes_length, command)
                self.queue_gather2actors[actor_index].put((command, "successfully send episodes"))

            elif command == "sample_reward":
                self.cache_and_send_data(self.cached_sample_reward, data, self.max_cached_sample_reward_length, command)
                self.queue_gather2actors[actor_index].put((command, "successfully send sample reward"))

            elif command == "eval_reward":
                self.cache_and_send_data(self.cached_eval_reward, data, self.max_cached_eval_reward_length, command)
                self.queue_gather2actors[actor_index].put((command, "successfully send eval reward"))

    def cache_and_send_data(self, cache_list: list, data_list: list, max_cache_list_length: int, command: str):
        cache_list.extend(data_list)
        if len(cache_list) > max_cache_list_length:
            logging.debug(connection.send_recv(self.server_conn, (command, cache_list)))
            cache_list.clear()


def _open_per_gather(gather_id, host, port, num_sample_actors, num_predict_actors, func, logger_file_path):
    set_process_logger(file_path=logger_file_path)  # windows need
    server_conn = connection.connect_socket_connection(host, port)
    logging.info(f"successfully connect to {host}:{port}! the gather {gather_id} is started!")
    Gather(gather_id, server_conn, num_sample_actors, num_predict_actors, func).run()


def open_gather(host: str, port: int, num_gathers: int,
                num_sample_actors_per_gather: int, num_predict_actors_per_gather: int = 0,
                func: Callable = None, logger_file_dir=None):
    mp.set_start_method("spawn")
    processes = []
    for i in range(num_gathers):
        p = mp.Process(name=f"gather_{i}", target=_open_per_gather,
                       args=(i, host, port, num_sample_actors_per_gather, num_predict_actors_per_gather,
                             func, logger_file_dir + f"gather_{i}.txt"),
                       daemon=False
                       )
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.close()


def create_actor(actor_index: int, queue_gather2actor, queue_actor2gather):
    raise NotImplementedError


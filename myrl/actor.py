import bz2
import os
import pickle

from myrl.agent import Agent
import myrl.connection as connection

from tensorboardX import SummaryWriter
from typing import Callable, Tuple, List, Dict, Union
import logging
import multiprocessing as mp

from collections import defaultdict

from myrl.utils import batchify, set_process_logger

__all__ = ["Actor", "ActorCreateBase", "open_gather"]


class Actor:
    def __init__(self, env, agent: Agent,
                 steps: int = 50,
                 get_full_episode: bool = False,
                 tensorboard_dir: str = None
                 ):
        """
        steps 和 get_full_episode 共同控制一次采样的长度

        :param env:   gym api
        :param agent: 接受batched np.array 作为输入, 返回action_info, 其中必须有action字段
        :param steps: 采用当前模型采样steps步
        :param get_full_episode: 若采样steps步之后，继续采样一定步数直到当前episode结束
        """
        self.env = env
        self.agent = agent

        self.steps = steps
        self.get_full_episode = get_full_episode

        self.num_episodes = 0

        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)

        # 初始化当前信息
        # self.hidden = None
        self.obs = self.env.reset()
        self.done = False
        self.current_episode_infos = defaultdict(lambda: 0)

        self.episodes_infos = defaultdict(list)

    def sample(self, model_id=None):
        # episode generation
        episode = []
        step = 0

        while step < self.steps or (not self.done and self.get_full_episode):

            moment = dict()
            moment['observation'] = self.obs

            # batch obs
            action_info = self.agent.sample(batchify([self.obs], unsqueeze=0))

            for key, value in action_info.items():
                moment[key] = value[0]

            self.obs, reward, self.done, info = self.env.step(moment['action'])
            step += 1

            moment["reward"] = reward
            moment["done"] = self.done

            episode.append(moment)

            self.current_episode_infos["steps"] += 1
            self.current_episode_infos["reward"] += reward

            if self.done:
                self._record_update_and_reset_when_done(info, model_id)

                if step >= self.steps:
                    break

        return episode

    def predict(self, model_id=None):
        step = 0
        while step < self.steps or (not self.done and self.get_full_episode):

            action_info = self.agent.predict(batchify([self.obs], unsqueeze=0))

            self.obs, reward, self.done, info = self.env.step(action_info['action'][0])

            step += 1

            self.current_episode_infos["steps"] += 1
            self.current_episode_infos["reward"] += reward

            if self.done:
                self._record_update_and_reset_when_done(info, model_id)

                if step >= self.steps:
                    break
        return

    def _record_update_and_reset_when_done(self, ending_info, model_id):
        """
        1. record according to num_episodes, current_episode_infos and ending_info
        """
        self.num_episodes += 1
        logging.info(f"num_episodes is : {self.num_episodes}, "
                     f"current episode_infos is : {self.current_episode_infos}, "
                     f"ending info is {ending_info}")

        if hasattr(self, "sw"):
            for key, value in self.current_episode_infos.items():
                self.sw.add_scalar(key, value, self.num_episodes)
        """
        2. update episodes_infos according to current_episode_infos and model_index
        """
        for key, value in self.current_episode_infos.items():
            self.episodes_infos[key].append(value)

        if model_id is not None:
            self.episodes_infos["model_id"].append(model_id)

        """
        reset obs, done and current_episode_infos
        """
        self.obs = self.env.reset()
        self.done = False
        self.current_episode_infos = defaultdict(lambda: 0)


"""
actor_client and gather communication protocol:
cmd           args                       explanation
model         None      
model         (model_index, weights) 
episode       (model_index, raw_episode)
sample_infos  Dict[metric, value_list]
eval_infos    Dict[metric, value_list]
"""


class ActorClient:
    def __init__(self,
                 actor_id: int,
                 role: str,  # Literal["sampler", "evaluator"]
                 actor: Actor,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 use_bz2: bool
                 ):

        assert(role in ["sampler", "evaluator"])

        self.actor_id = actor_id
        self.role = role
        self.actor = actor
        self.queue_gather2actor = queue_gather2actor
        self.queue_actor2gather = queue_actor2gather
        self.use_bz2 = use_bz2

    def run(self):
        if self.role == "sampler":
            self._run_sampler()

        elif self.role == "evaluator":
            self._run_evaluator()

    def _run_evaluator(self):
        """
        data_type: (id, cmd, data or args)
        """
        send_infos_length = 5
        while True:
            model_id = self._request_weights_and_set()

            self.actor.predict(model_id)

            if len(self.actor.episodes_infos["model_id"]) >= send_infos_length:
                cmd, msg = connection.send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                      self.queue_gather2actor,
                                                                      self.actor_id,
                                                                      ("eval_infos", self.actor.episodes_infos))
                self.actor.episodes_infos.clear()
                logging.debug(f"{cmd}_{msg}")

    def _run_sampler(self):
        send_infos_length = 5
        while True:
            model_id = self._request_weights_and_set()

            episode = self.actor.sample(model_id)

            if self.use_bz2:
                episode = bz2.compress(pickle.dumps(episode))

            cmd, msg = connection.send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                  self.queue_gather2actor,
                                                                  self.actor_id,
                                                                  ("episodes", (model_id, episode)))

            logging.debug(f"{cmd}_{msg}")  # cmd, msg

            if len(self.actor.episodes_infos["model_id"]) >= send_infos_length:
                cmd, msg = connection.send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                      self.queue_gather2actor,
                                                                      self.actor_id,
                                                                      ("sample_infos", self.actor.episodes_infos))
                self.actor.episodes_infos.clear()
                logging.debug(f"{cmd}_{msg}")

    def _request_weights_and_set(self):
        """
        1. request weights
        """
        cmd, (model_id, weights) = connection.send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                              self.queue_gather2actor,
                                                                              self.actor_id, ("model", None))
        logging.debug(f"successfully request model weights {model_id}")
        """
        2. decompress? and set weights
        """
        if self.use_bz2:
            weights = pickle.loads(bz2.decompress(weights))

        self.actor.agent.set_weights(weights)

        return model_id


class ActorCreateBase:
    def __init__(self, logger_file_dir: str = None, steps: int = None):
        """
        the class of create actor and run sampling or predicting.
        the user should inherit this class and implement create_env_and_agent.
        :param logger_file_dir: the logger file directory
        :param steps: the sample steps of each actor, when the actor role is evaluator, this parameter will be ignored
        """
        self.logger_file_dir = logger_file_dir
        self.logger_file_path = None
        self.steps = steps

    def __call__(self, infos: tuple[int, int, str, bool], queue_gather2actor: mp.Queue, queue_actor2gather: mp.Queue):
        gather_id, actor_id, actor_role, use_bz2 = infos

        assert (actor_role in ["sampler", "evaluator"])

        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"gather_{gather_id}_actor_{actor_id}.txt")

        set_process_logger(file_path=self.logger_file_path)

        env, agent = self.create_env_and_agent(gather_id, actor_id)

        if actor_role == "sampler":
            actor = Actor(env, agent, steps=self.steps, get_full_episode=False)  # 每次采样step步
            actor_client = ActorClient(actor_id=actor_id, role=actor_role, actor=actor,
                                       queue_gather2actor=queue_gather2actor, queue_actor2gather=queue_actor2gather,
                                       use_bz2=use_bz2)
            actor_client.run()
        elif actor_role == "evaluator":
            actor = Actor(env, agent, steps=-1, get_full_episode=True)  # 每次采样一个episode
            actor_client = ActorClient(actor_id=actor_id, role=actor_role, actor=actor,
                                       queue_gather2actor=queue_gather2actor, queue_actor2gather=queue_actor2gather,
                                       use_bz2=use_bz2)
            actor_client.run()

    def create_env_and_agent(self, gather_id: int, actor_id: int):
        raise NotImplementedError


class Gather:
    def __init__(self,
                 gather_id: int,
                 memory_server_conn: connection.PickledConnection,
                 league_conn: connection.PickledConnection,
                 num_actors: int,
                 actor_role: str,  # Literal["sampler", "evaluator"]
                 func: ActorCreateBase,
                 use_bz2=True):
        super().__init__()
        self.gather_id = gather_id

        self.memory_server_conn = memory_server_conn  # send episodes and sample_infos
        self.league_conn = league_conn  # request model and send eval_infos

        self.num_actors = num_actors
        self.actor_role = actor_role
        self.use_bz2 = use_bz2

        """
        model part
        max_num_sent_weights 表示每个weights最多send的次数
        """
        self.model_id = None
        self.weights = None
        self.num_sent_weights = 0
        self.max_num_sent_weights = self.num_actors

        # sample infos or eval infos
        self.infos = defaultdict(list)
        self.max_infos_length = self.num_actors * 5

        # sample episodes
        self.episodes = []
        self.max_episodes_length = self.num_actors

        self.queue_gather2actors = []
        self.queue_actor2gather = mp.Queue(maxsize=self.num_actors)

        for i in range(self.num_actors):
            self.queue_gather2actors.append(mp.Queue(maxsize=1))
            mp.Process(target=func, args=((gather_id, i, self.actor_role, self.use_bz2),
                                          self.queue_gather2actors[i],
                                          self.queue_actor2gather),
                       name=f"gather_{gather_id}_actor_{i}", daemon=True).start()

    def run(self):

        self._request_weights_and_reset()

        while True:

            actor_id, (command, data) = self.queue_actor2gather.get()
            actor_id: int

            logging.debug(f"actor_index : {actor_id}, command : {command}")

            if command == "model":

                if self.num_sent_weights > self.max_num_sent_weights:
                    self._request_weights_and_reset()

                connection.send(self.queue_gather2actors[actor_id], (command, (self.model_id, self.weights)))
                self.num_sent_weights += 1

            elif command == "episodes":

                self.episodes.append(data)

                if len(self.episodes) >= self.max_episodes_length:
                    logging.debug(connection.send_recv(self.memory_server_conn, ("episodes", self.episodes)))
                    self.episodes.clear()

                connection.send(self.queue_gather2actors[actor_id], (command, "successfully send episodes"))

            elif command == "sample_infos":
                for key, value in data.items():
                    self.infos[key].extend(value)

                if len(self.infos["model_id"]) >= self.max_infos_length:
                    logging.debug(connection.send_recv(self.memory_server_conn, ("sample_infos", self.infos)))
                    self.infos.clear()

                connection.send(self.queue_gather2actors[actor_id], (command, "successfully send sample_infos"))

            elif command == "eval_infos":

                for key, value in data.items():
                    self.infos[key].extend(value)

                if len(self.infos["model_id"]) >= self.max_infos_length:
                    logging.debug(connection.send_recv(self.league_conn, ("eval_infos", self.infos)))
                    self.infos.clear()

                connection.send(self.queue_gather2actors[actor_id], (command, "successfully send eval_infos"))

    def _request_weights_and_reset(self):
        if self.actor_role == "sampler":
            _, (model_id, weights) = connection.send_recv(self.league_conn, ("model", "latest"))
            self.model_id = model_id
            self.weights = weights
            self.num_sent_weights = 0

        elif self.actor_role == "evaluator":
            _, (model_id, weights) = connection.send_recv(self.league_conn, ("model", -1))
            self.model_id = model_id
            self.weights = weights
            self.num_sent_weights = 0

        logging.debug("successfully request weights and reset !")


def _open_per_gather(gather_id: int,
                     memory_server_address: Tuple[str, int],
                     league_address: Tuple[str, int],
                     num_actors: int, actor_role: str,
                     func: ActorCreateBase,
                     use_bz2: bool,
                     logger_file_path: str):

    set_process_logger(file_path=logger_file_path)
    memory_server_conn = connection.connect_socket_connection(*memory_server_address)
    league_conn = connection.connect_socket_connection(*league_address)

    logging.info(f"successfully connected! the gather {gather_id} is started!")
    Gather(gather_id=gather_id, memory_server_conn=memory_server_conn, league_conn=league_conn,
           num_actors=num_actors, actor_role=actor_role, func=func, use_bz2=use_bz2).run()


def open_gather(memory_server_address: Tuple[str, int],
                league_address: Tuple[str, int],
                num_gathers: int,
                num_actors: Union[int, List[int], Tuple[int]],
                actor_roles: Union[str, List[str], Tuple[str]],
                func: ActorCreateBase,
                use_bz2: bool = True,
                logger_file_dir=None):

    if isinstance(num_actors, int):
        num_actors = [num_actors] * num_gathers

    if isinstance(actor_roles, str):
        actor_roles = [actor_roles] * num_gathers

    mp.set_start_method("spawn")
    processes = []
    for i in range(num_gathers):
        p = mp.Process(name=f"gather_{i}",
                       target=_open_per_gather,
                       args=(i, memory_server_address, league_address,
                             num_actors[i], actor_roles[i],
                             func, use_bz2, os.path.join(logger_file_dir, f"gather_{i}.txt")), daemon=False)
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.close()




import bz2
import os
import pickle
import typing

from myrl.agent import Agent
import myrl.connection as connection

from tensorboardX import SummaryWriter
from typing import Optional, Tuple, List, Dict, Union, Any
import logging
import multiprocessing as mp

from collections import defaultdict

from myrl.utils import batchify, set_process_logger

__all__ = ["Actor", "ActorCreateBase", "open_gather"]


class Actor:
    def __init__(self, env, agent: Agent,
                 num_steps: int = 0,
                 num_episodes: int = 0,
                 get_full_episode: bool = False,
                 tensorboard_dir: str = None
                 ):
        """
        一次采样至少num_steps步，
        至少采样num_episodes个完整的episode,在steps步之后，
        若get_full_episode等于True,继续采样一个完整的episode

        :param env:   gym api
        :param agent: 接受batched np.array 作为输入, 返回action_info, 其中必须有action字段
        :param num_steps: 采用当前模型采样num_steps步
        :param num_episodes: 采样num_episodes个完整的episode
        :param get_full_episode: 若采样steps步之后，继续采样一定步数直到当前episode结束
        """
        self.env = env
        self.agent = agent

        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.get_full_episode = get_full_episode

        self.total_episodes = 0

        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)

        # 初始化当前信息
        # self.hidden = None
        self.obs = self.env.reset()
        self.done = False
        self.current_episode_infos = defaultdict(lambda: 0)

        self.episodes_infos = defaultdict(list)

    def sample(self, model_id=None) -> List[Dict[str, Any]]:
        # episode generation
        num_episodes = 0
        episode = []
        step = 0

        while step < self.num_steps or num_episodes < self.num_episodes or (not self.done and self.get_full_episode):

            moment = dict()
            moment['observation'] = self.obs

            # batch obs
            action_info = self.agent.sample(batchify([self.obs], unsqueeze=0))

            for key, value in action_info.items():
                moment[key] = value[0]

            self.obs, reward_infos, self.done, info = self.env.step(moment['action'])
            step += 1

            moment["reward_infos"] = reward_infos
            moment["done"] = self.done

            episode.append(moment)

            self.current_episode_infos["steps"] += 1
            for key, value in reward_infos.items():
                self.current_episode_infos[key] += value

            if self.done:
                self._record_update_and_reset_when_done(info, model_id)

                num_episodes += 1
                if step >= self.num_steps and num_episodes >= self.num_episodes:
                    break

        return episode

    def predict(self, model_id=None) -> None:
        num_episodes = 0
        step = 0

        while step < self.num_steps or num_episodes < self.num_episodes or (not self.done and self.get_full_episode):

            action_info = self.agent.predict(batchify([self.obs], unsqueeze=0))

            self.obs, reward_infos, self.done, info = self.env.step(action_info['action'][0])

            step += 1

            self.current_episode_infos["steps"] += 1
            for key, value in reward_infos.items():
                self.current_episode_infos[key] += value

            if self.done:
                self._record_update_and_reset_when_done(info, model_id)

                num_episodes += 1
                if step >= self.num_steps and num_episodes >= self.num_episodes:
                    break
        return

    def _record_update_and_reset_when_done(self, ending_info, model_id=None):
        """
        1. record according to num_episodes, current_episode_infos and ending_info
        """
        self.total_episodes += 1
        logging.info(f"num_episodes is : {self.total_episodes}, "
                     f"current episode_infos is : {self.current_episode_infos}, "
                     f"ending info is {ending_info}")

        if hasattr(self, "sw"):
            for key, value in self.current_episode_infos.items():
                self.sw.add_scalar(key, value, self.total_episodes)
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


#%%
"""
actor_client and gather communication protocol

actor_client send data format (id:int, (cmd:str, data:Any))
actor_client receive data format (cmd:str, data:Any)

actor_client and gather communication data type:
cmd           model                       explanation
model         None      
model         (model_index, weights) 
episodes      (model_index, raw_episode)
sample_infos  Dict[metric, value_list]
eval_infos    Dict[metric, value_list]


actor_client logic:
  request and receive weights (uncompress?)
  sample or predict (send episodes when sample)
  send episodes_infos when necessary.

gather logic:
   request and receive weights from league
   send weight num_actor times
   cache episodes and episodes_infos and send to server
"""


def send_with_sender_id_and_receive(queue_sender: mp.Queue, queue_receiver: mp.Queue,
                                    sender_id: int, data: Tuple[str, Any],
                                    block: bool = True, timeout: float = None) -> Tuple[str, Any]:
    queue_sender.put((sender_id, data), block=block, timeout=timeout)
    cmd, data = queue_receiver.get()
    return cmd, data


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
                cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                           self.queue_gather2actor,
                                                           self.actor_id,
                                                           ("eval_infos", self.actor.episodes_infos))
                self.actor.episodes_infos.clear()
                logging.debug(f"{cmd} response: {msg}")

    def _run_sampler(self):
        send_infos_length = 5
        while True:
            model_id = self._request_weights_and_set()

            episode = self.actor.sample(model_id)

            if self.use_bz2:
                episode = bz2.compress(pickle.dumps(episode))

            cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                       self.queue_gather2actor,
                                                       self.actor_id,
                                                       ("episodes", (model_id, episode)))

            logging.debug(f"{cmd} response: {msg}")  # cmd, msg

            if len(self.actor.episodes_infos["model_id"]) >= send_infos_length:
                cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                           self.queue_gather2actor,
                                                           self.actor_id,
                                                           ("sample_infos", self.actor.episodes_infos))
                self.actor.episodes_infos.clear()
                logging.debug(f"{cmd} response: {msg}")

    def _request_weights_and_set(self):
        """
        1. request weights
        """
        cmd, (model_id, weights) = send_with_sender_id_and_receive(self.queue_actor2gather,
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
#%%


"""
API Class
"""


class ActorCreateBase:
    def __init__(self, num_steps: int = 50, num_episodes: int = 0, get_full_episode: bool = False,
                 logger_file_dir: str = None,):
        """
        the class of create actor and run sampling or predicting.
        the user should inherit this class and implement create_env_and_agent.
        :param logger_file_dir: the logger file directory
        :param num_steps: the sample steps of each actor, when the actor role is evaluator,
                          this parameter will be ignored
        :param num_episodes: the episodes num of each actor, when the actor role is evaluator,
                             this parameter will be ignored
        :param get_full_episode: the flag of whether to get full episode of each actor,
                                 when the actor role is evaluator,
                                 this parameter will be ignored
        """

        self.logger_file_dir = logger_file_dir
        self.logger_file_path = None

        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.get_full_episode = get_full_episode

    def __call__(self, infos: Tuple[int, int, str, bool], queue_gather2actor: mp.Queue, queue_actor2gather: mp.Queue):
        gather_id, actor_id, actor_role, use_bz2 = infos

        assert (actor_role in ["sampler", "evaluator"])

        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"gather_{gather_id}_actor_{actor_id}.txt")

        set_process_logger(file_path=self.logger_file_path)

        env, agent = self.create_env_and_agent(gather_id, actor_id)

        if actor_role == "sampler":
            actor = Actor(env, agent,
                          num_steps=self.num_steps,
                          num_episodes=self.num_episodes,
                          get_full_episode=self.get_full_episode)

            actor_client = ActorClient(actor_id=actor_id, role=actor_role, actor=actor,
                                       queue_gather2actor=queue_gather2actor, queue_actor2gather=queue_actor2gather,
                                       use_bz2=use_bz2)
            actor_client.run()

        elif actor_role == "evaluator":
            actor = Actor(env, agent,
                          num_steps=0,
                          num_episodes=5,
                          get_full_episode=True)  # 每次采样5个episode

            actor_client = ActorClient(actor_id=actor_id, role=actor_role, actor=actor,
                                       queue_gather2actor=queue_gather2actor, queue_actor2gather=queue_actor2gather,
                                       use_bz2=use_bz2)
            actor_client.run()

    def create_env_and_agent(self, gather_id: int, actor_id: int):
        raise NotImplementedError


#%%

def send(queue_sender: mp.Queue, data: Tuple[str, Any],
         block: bool = True, timeout: float = None):
    queue_sender.put(data, block=block, timeout=timeout)


class Gather:
    def __init__(self,
                 gather_id: int,
                 num_actors: int,
                 actor_role: str,  # Literal["sampler", "evaluator"]
                 league_conn: connection.PickledConnection,
                 memory_server_conn: Optional[connection.PickledConnection],
                 func: ActorCreateBase,
                 use_bz2=True):
        super().__init__()

        assert(actor_role in ["sampler", "evaluator"])
        if actor_role == "sampler" and memory_server_conn is None:
            raise ValueError("sampler actor must designate memory_server_conn that used to receive episodes")

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

                if self.num_sent_weights >= self.max_num_sent_weights:
                    self._request_weights_and_reset()

                send(self.queue_gather2actors[actor_id], (command, (self.model_id, self.weights)))
                self.num_sent_weights += 1

            elif command == "episodes":

                self.episodes.append(data)

                if len(self.episodes) >= self.max_episodes_length:
                    logging.debug(connection.send_recv(self.memory_server_conn, ("episodes", self.episodes)))
                    self.episodes.clear()

                send(self.queue_gather2actors[actor_id], (command, "successfully receive episodes"))

            elif command == "sample_infos":
                for key, value in data.items():
                    self.infos[key].extend(value)

                if len(self.infos["model_id"]) >= self.max_infos_length:
                    logging.debug(connection.send_recv(self.memory_server_conn, ("sample_infos", self.infos)))
                    self.infos.clear()

                send(self.queue_gather2actors[actor_id], (command, "successfully receive sample_infos"))

            elif command == "eval_infos":

                for key, value in data.items():
                    self.infos[key].extend(value)

                if len(self.infos["model_id"]) >= self.max_infos_length:
                    logging.debug(connection.send_recv(self.league_conn, ("eval_infos", self.infos)))
                    self.infos.clear()

                send(self.queue_gather2actors[actor_id], (command, "successfully receive eval_infos"))

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
                     memory_server_address: Optional[Tuple[str, int]],
                     league_address: Tuple[str, int],
                     num_actors: int, actor_role: str,
                     func: ActorCreateBase,
                     use_bz2: bool,
                     logger_file_path: str):

    assert (actor_role in ["sampler", "evaluator"])
    if actor_role == "sampler" and memory_server_address is None:
        raise ValueError("sampler actor must designate memory_server_address that used to receive episodes")

    set_process_logger(file_path=logger_file_path)

    if memory_server_address is not None:
        memory_server_conn = connection.connect_socket_connection(*memory_server_address)
    else:
        memory_server_conn = None

    league_conn = connection.connect_socket_connection(*league_address)

    logging.info(f"successfully connected! the gather {gather_id} is starting!")

    gather = Gather(gather_id=gather_id, memory_server_conn=memory_server_conn, league_conn=league_conn,
                    num_actors=num_actors, actor_role=actor_role, func=func, use_bz2=use_bz2)

    logging.info(f"gather {gather_id} is started!")

    gather.run()


AddrType = Tuple[str, int]


def open_gather(num_gathers: int,
                memory_server_address: Union[AddrType, List[AddrType], Tuple[AddrType]],
                league_address: Union[AddrType, List[AddrType], Tuple[AddrType]],
                num_actors: Union[int, List[int], Tuple[int]],
                actor_roles: Union[str, List[str], Tuple[str]],
                func: ActorCreateBase,
                use_bz2: bool = True,
                logger_file_dir=None):
    """

    :param num_gathers:
    :param memory_server_address:
    :param league_address:
    :param num_actors:
    :param actor_roles:
    :param func:
    :param use_bz2:
    :param logger_file_dir:
    :return:
    """

    if isinstance(memory_server_address[0], str):
        memory_server_address = [memory_server_address] * num_gathers

    if isinstance(league_address[0], str):
        league_address = [league_address] * num_gathers

    if isinstance(num_actors, int):
        num_actors = [num_actors] * num_gathers

    if isinstance(actor_roles, str):
        actor_roles = [actor_roles] * num_gathers

    mp.set_start_method("spawn")
    processes = []

    for i in range(num_gathers):
        if logger_file_dir is not None:
            logger_file_path = os.path.join(logger_file_dir, f"gather_{i}.txt")
        else:
            logger_file_path = None
        p = mp.Process(name=f"gather_{i}",
                       target=_open_per_gather,
                       args=(i, memory_server_address[i], league_address[i],
                             num_actors[i], actor_roles[i],
                             func, use_bz2, logger_file_path), daemon=False)
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.close()




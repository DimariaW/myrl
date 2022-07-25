import logging
import torch
import queue
import multiprocessing as mp
import threading

from typing import Union, List, Dict
from tensorboardX import SummaryWriter

import myrl.connection as connection
import myrl.utils as utils

import pickle
import bz2


class MemoryReplayBase:
    """
       MemoryReplayServer is the data structure that caches the episode sent from actors
    """
    def __init__(self, queue_receiver: mp.Queue):
        """
        at least have queue_receiver
        :param queue_receiver: the queue that caches batched np.ndarray
        """
        self.queue_receiver = queue_receiver

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]) -> None:
        """
        the api that caches the episode
        :param episodes: list of episode,
               episode itself is a list of dict,
               dict contains obs, action, reward, done
        :return: None
        """
        raise NotImplementedError

    def start(self):
        """
        the logic that batchifies the cached raw episodes
        :return: None
        """
        raise NotImplementedError

    def stop(self):
        pass


def make_batch(episodes: List[List]):  # batch_size*time_step
    episodes = [utils.batchify(episode, unsqueeze=0) for episode in episodes]
    return utils.batchify(episodes, unsqueeze=0)


def decompress_and_make_batch(episodes: List):
    episodes = [utils.batchify(pickle.loads(bz2.decompress(episode)), unsqueeze=0) for episode in episodes]
    return utils.batchify(episodes, unsqueeze=0)


class TrajList(MemoryReplayBase):
    """
    used for a2c algorithm
    """
    def __init__(self, queue_receiver):
        super().__init__(queue_receiver)
        self.episode_list = []
        self.num_cashed = 0
        self.num_sent = 0

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]):
        if isinstance(episodes[0], list):
            self.episode_list.extend(episodes)
            self.num_cashed += len(episodes)
        elif isinstance(episodes[0], dict):
            self.episode_list.append(episodes)
            self.num_cashed += 1
        else:
            raise TypeError("episodes must be list of traj or traj, traj itself is a list of dict")
        logging.debug(f"total cached data num is {self.num_cashed}")

    def start(self):
        batch = make_batch(self.episode_list)
        self.queue_receiver.put((False, batch))  # False 表示发送端没有停止发送
        self.num_sent += 1
        logging.debug(f"total sent data num is {self.num_sent}")
        self.episode_list.clear()

    def stop(self):
        self.queue_receiver.put((True, None))
        logging.debug(f'successfully stop sending data')


class TrajQueue(MemoryReplayBase):
    def __init__(self,
                 maxlen: int,
                 queue_receiver: mp.Queue,
                 batch_size=64,
                 use_bz2: bool = True):
        super().__init__(queue_receiver)
        self.episode_queue = queue.Queue(maxsize=maxlen)
        self.batch_size = batch_size
        self.num_cached = 0
        self.is_stop = False
        self.use_bz2 = use_bz2

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]):
        if isinstance(episodes[0], dict):
            while True:
                try:
                    self.episode_queue.put(episodes, timeout=0.1)
                    break
                except queue.Full:
                    logging.debug("the queue is full")
            self.num_cached += 1
            logging.debug(f"total cashed data num is {self.num_cached}")

        elif isinstance(episodes[0], list):
            for episode in episodes:
                while True:
                    try:
                        self.episode_queue.put(episode, timeout=0.1)
                        break
                    except queue.Full:
                        logging.debug("the queue is full")
                self.num_cached += 1
            logging.debug(f"total cashed data num is {self.num_cached}")

    def send_raw_batch(self):
        while True:
            raw_batch = []
            for _ in range(self.batch_size):
                while True:
                    try:
                        raw_batch.append(self.episode_queue.get(timeout=0.1))
                        break
                    except queue.Empty:
                        if self.is_stop:
                            return
            yield raw_batch

    def start(self):
        threading.Thread(target=self._make_batch, args=(), name="batch_maker", daemon=True).start()

    @utils.wrap_traceback
    def _make_batch(self):
        num = 0
        send_generator = self.send_raw_batch()
        try:
            while True:
                if self.use_bz2:
                    batched = decompress_and_make_batch(next(send_generator))
                else:
                    batched = make_batch(next(send_generator))
                self.queue_receiver.put((False, batched))
                num += 1
                logging.debug(f"successfully make and send batch num: {num}")
        except StopIteration:
            self.queue_receiver.put((True, None))
            logging.debug(f"successfully stop send!")

    def stop(self):
        self.is_stop = True


class TrajQueueMP(TrajQueue):
    def __init__(self,
                 maxlen: int,
                 queue_receiver: mp.Queue,
                 batch_size: int = 64,
                 use_bz2: bool = True,
                 num_batch_maker: int = 2,
                 logger_file_dir: str = None,
                 ):
        super().__init__(maxlen, queue_receiver, batch_size, use_bz2)

        self.batch_maker = connection.MultiProcessJobExecutors(func=make_batch if not use_bz2
                                                               else decompress_and_make_batch,

                                                               send_generator=self.send_raw_batch(),
                                                               num=num_batch_maker,
                                                               queue_receiver=self.queue_receiver,
                                                               name_prefix="batch_maker",
                                                               logger_file_dir=logger_file_dir)

    def start(self):
        self.batch_maker.start()


class MemoryReplayServer:
    def __init__(self,
                 memory_replay: MemoryReplayBase,
                 port: int,
                 actor_num=None,
                 tensorboard_dir=None
                 ):

        self.actor_communicator = connection.QueueCommunicator(port, actor_num)
        self.memory_replay = memory_replay

        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)

        self.num_received_sample_episodes = 0
        self.num_received_eval_episodes = 0
        self.actor_num = actor_num

    def run(self):
        logging.info("start server to receive episodes that generated by actor")
        self.actor_communicator.run()
        self.memory_replay.start()
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            logging.debug(cmd)
            if cmd == "episode":
                self.memory_replay.cache(data)
                self.actor_communicator.send(conn, (cmd, "successfully sent episodes"))

            elif cmd == "sample_reward":
                self._process_sample_rewards(conn, cmd, data)

            elif cmd == "eval_reward":
                self._process_eval_rewards(conn, cmd, data)

    def run_sync(self):
        self.actor_communicator.run_sync()

        conns = []
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()

            logging.debug(cmd)

            if cmd == "episodes":
                self.memory_replay.cache(data)
                conns.append(conn)

                if len(conns) == self.actor_num:
                    self.memory_replay.start()
                    for conn in conns:
                        self.actor_communicator.send(conn, (cmd, "successfully sent episodes"))
                    conns = []

            elif cmd == "sample_rewards":
                self._process_sample_rewards(conn, cmd, data)

            elif cmd == "eval_rewards":
                self._process_eval_rewards(conn, cmd, data)

    def _process_sample_rewards(self, conn, cmd, rewards):
        assert(cmd == "sample_reward")
        for reward in rewards:
            self.num_received_sample_episodes += 1
            self.sw.add_scalar(tag="sample_reward", scalar_value=reward, global_step=self.num_received_sample_episodes)
        self.actor_communicator.send(conn, (cmd, "successfully sent sample rewards"))

    def _process_eval_rewards(self, conn, cmd, rewards):
        assert (cmd == "eval_reward")
        for reward in rewards:
            self.num_received_eval_episodes += 1
            self.sw.add_scalar(tag="eval_reward", scalar_value=reward, global_step=self.num_received_eval_episodes)
        self.actor_communicator.send(conn, (cmd, "successfully sent eval rewards"))


class TensorReceiver(connection.Receiver):
    def __init__(self,
                 queue_receiver: mp.Queue,
                 num_sender,
                 device=torch.device("cpu")):
        super().__init__(queue_receiver, num_sender=num_sender)
        self.device = device

    def recall(self):
        logging.debug(f"current queue receiver size is : {self.queue_receiver.qsize()}")
        batch = self.recv()
        return utils.to_tensor(batch, unsqueeze=None, device=self.device)



"""
class TrajQueueMP(MemoryReplayBase):
    def __init__(self,
                 maxlen: int,
                 device=torch.device("cpu"),
                 batch_size: int = 64,
                 num_batch_maker: int = 2,
                 logger_file_path: str = None,
                 file_level=logging.DEBUG
                 ):
        self.episodes = queue.Queue(maxsize=maxlen)
        self.device = device
        self.batch_size = batch_size

        self.num_cached = 0

        self.batch_maker = connection.MultiProcessJobExecutors(func=make_batch, send_generator=self.send_raw_batch(),
                                                      postprocess=self.post_process,
                                                      num=num_batch_maker,
                                                      buffer_length=1,
                                                      name_prefix="batch_maker",
                                                      logger_file_path=logger_file_path,
                                                      file_level=file_level)

    def cache(self, episodes: List[List]):
        for episode in episodes:
            while True:
                try:
                    self.episodes.put(episode, timeout=0.1)
                    self.num_cached += 1
                    logging.debug(f"successfully cached episode num {self.num_cached}")
                    break
                except queue.Full:
                    logging.critical(" generate rate is larger than consumer")

    def recall(self):
        return self.batch_maker.recv()

    def send_raw_batch(self):
        while True:
            yield [self.episodes.get() for _ in range(self.batch_size)]

    def start(self):
        self.batch_maker.start()

    def post_process(self, batch):
        return utils.to_tensor(batch, unsqueeze=None, device=self.device)
"""




"""
class MultiProcessBatcher:
    def __init__(self,
                 maxlen: int,
                 device=torch.device("cpu"),
                 batch_size: int = 192,
                 forward_steps: int = 64,
                 num_batch_maker: int = 2,
                 use_queue: bool = True,
                 logger_file_path: str = "./log/log.txt",
                 file_level=logging.DEBUG
                 ):
        self.episodes = deque(maxlen=maxlen)
        self.device = device
        self.batch_size = batch_size
        self.forward_steps = forward_steps

        self.num_cached = 0

        if not use_queue:
            self.batch_maker = connection.MultiProcessJobExecutors(func=make_batch, send_generator=self.send_raw_batch(),
                                                        post_process=self.post_process,
                                                        num=num_batch_maker, buffer_length=1 + 8//num_batch_maker,
                                                        num_receivers=1,
                                                        name_prefix="batch_maker",
                                                        logger_file_path=logger_file_path,
                                                        file_level=file_level)
        else:
            self.batch_maker = connection.MultiProcessJobExecutors(func=make_batch, send_generator=self.send_raw_batch(),
                                                          postprocess=self.post_process,
                                                          num=num_batch_maker, buffer_length=1 + 8//num_batch_maker,
                                                          name_prefix="batch_maker",
                                                          logger_file_path=logger_file_path,
                                                          file_level=file_level)

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]):
        if isinstance(episodes[0], list):
            self.episodes.extend(episodes)
            self.num_cached += len(episodes)
            logging.debug(f"total cached episodes is {self.num_cached}")
        elif isinstance(episodes[0], dict):
            self.episodes.append(episodes)
            self.num_cached += 1
            logging.debug(f"total cached episodes is {self.num_cached}")
        else:
            raise TypeError("episodes must be list of traj or traj, traj itself is a list of dict")

    def recall(self):
        return self.batch_maker.recv()

    def send_raw_batch(self):
        while True:
            yield [self.send_a_sample() for _ in range(self.batch_size)]

    def send_a_sample(self):
        while True:
            ep_idx = random.randrange(len(self.episodes))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.episodes.maxlen
            if random.random() < accept_rate:
                ep = self.episodes[ep_idx]
                st = random.randrange(
                    1 + max(0, len(ep) - self.forward_steps))  # change start turn by sequence length
                return ep[st:st+self.forward_steps]

    def start(self):
        self.batch_maker.start()

    def post_process(self, batch):
        return utils.to_tensor(batch, unsqueeze=None, device=self.device)

    def __len__(self):
        return len(self.episodes)
"""








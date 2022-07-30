import os
import multiprocessing as mp
import functools
import torch
from typing import List, Tuple, Union

import myrl.utils as utils
from myrl.connection import Receiver

from myrl.memory import MemoryServer, TrajList, TrajQueue, TrajQueueMP

from myrl.actor import ActorCreateBase, open_gather

__all__ = ["ActorCreateBase", "MemoryMainBase", "LearnerMainBase", "LeagueMainBase", "open_gather", "train_main"]


class MainBase:
    def __init__(self, name, logger_file_dir=None):
        """
        set logger file path to "logger_file_dir/name.txt"
        :param name: the name of file
        :param logger_file_dir: the logger file directory
        """
        self.logger_file_dir = logger_file_dir
        self.logger_file_path = None
        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"{name}.txt")


class MemoryMainBase(MainBase):
    def __init__(self, port: int, logger_file_dir=None):
        super().__init__("memory", logger_file_dir)
        self.port = port

    def __call__(self, queue_sender: mp.Queue):
        """
        自此函数中实例化MemoryServer对象，处理actor收集的数据.

        the train function will use this function like bellowing:

        import multiprocessing as mp
        queue_sender = mp.Queue(maxsize=1)
        mp.Process(target=actor_server_main, args=(sender,), daemon=False, name="actor_server").start()

        :param queue_sender: used to cache data generated
        :return: None
        """
        utils.set_process_logger(file_path=self.logger_file_path)
        self.main(queue_sender)

    def main(self, queue_sender: mp.Queue):
        raise NotImplementedError


class LearnerMainBase(MainBase):
    def __init__(self, logger_file_dir=None):
        super().__init__("learner", logger_file_dir)

    def __call__(self, queue_receiver: mp.Queue, queue_senders: Union[List[mp.Queue], Tuple[mp.Queue]]):
        """
        在此函数中实例化myrl.Algorithm的子类， 并且调用run 函数运行.

        the train function will use this function like bellowing:

        import multiprocessing as mp
        queue_receiver = mp.Queue(maxsize=1)
        queue_send = mp.Queue(maxsize=1)
        mp.Process(target=learner_main, args=(queue_receiver,queue_receiver), daemon=False, name="learner_main").start()

        :param queue_receiver: used to receiver data
        :param queue_sender: used to send model_weights
        :return: None
        """
        utils.set_process_logger(file_path=self.logger_file_path)
        self.main(queue_receiver, queue_senders)

    def main(self, queue_receiver: mp.Queue, qqueue_senders: Union[List[mp.Queue], Tuple[mp.Queue]]):
        raise NotImplementedError

    @staticmethod
    def create_receiver(queue_receiver: mp.Queue, device, num_sender=-1):
        to_device = functools.partial(utils.to_tensor, device=device, unsqueeze=None)
        return Receiver(queue_receiver, num_sender, postprocess=lambda data: (data[0], to_device(data[1])))


class LeagueMainBase(MainBase):
    def __init__(self, port: int, logger_file_dir=None):
        super().__init__("league", logger_file_dir)
        self.port = port

    def __call__(self, queue_receiver: mp.Queue):
        """
           the process to manage model weights.

           the train function will use this function like bellowing:

           import multiprocessing as mp
           queue_send = mp.Queue(maxsize=1)
           mp.Process(target=learner_main, args=(queue_send), daemon=False, name="league_main").start()

           :param queue_receiver: the queue to send model weights
           :return: None
        """
        utils.set_process_logger(file_path=self.logger_file_path)
        self.main(queue_receiver)

    def main(self, queue_receiver: mp.Queue):
        raise NotImplementedError

    @staticmethod
    def create_receiver(queue_receiver: mp.Queue, num_sender=-1):
        return Receiver(queue_receiver, num_sender=num_sender, postprocess=None)


def train_main(learner_main: LearnerMainBase,
               memory_mains: Union[List[MemoryMainBase], Tuple[MemoryMainBase]],
               league_mains: Union[List[LeagueMainBase], Tuple[LeagueMainBase]],
               memory_buffer_length=1):  # receiver and sender

    mp.set_start_method("spawn")

    queue_receiver = mp.Queue(maxsize=memory_buffer_length)  # receiver batched tensor, when on policy, this can be set to 1

    queue_senders = []
    for _ in range(len(league_mains)):
        queue_senders.append(mp.Queue(maxsize=1))  # the queue to send the newest data

    learner_process = mp.Process(target=learner_main, args=(queue_receiver, queue_senders),
                                 daemon=False, name="learner_main")
    learner_process.start()

    league_processes = []
    for i, league_main in enumerate(league_mains):
        league_process = mp.Process(target=league_main, args=(queue_senders[i],),
                                    daemon=False, name="league_main")
        league_process.start()
        league_processes.append(league_process)

    memory_processes = []
    for i, memory_main in enumerate(memory_mains):
        memory_process = mp.Process(target=memory_main, args=(queue_receiver,),
                                    daemon=False, name=f"memory_main_{i}")
        memory_process.start()
        memory_processes.append(memory_process)

    try:
        learner_process.join()
        for league_process in league_processes:
            league_process.join()
        for memory_process in memory_processes:
            memory_process.join()
    finally:
        learner_process.close()
        for league_process in league_processes:
            league_process.close()
        for memory_process in memory_processes:
            memory_process.close()









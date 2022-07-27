import os
import multiprocessing as mp
import torch
from typing import List, Tuple, Union

import myrl.utils as utils
from myrl.connection import TensorReceiver, Receiver

from myrl.actor import ActorCreateBase, open_gather

__all__ = ["ActorCreateBase", "MemoryMainBase", "LearnerMainBase", "LeagueMainBase", "open_gather", "train_main"]


class MainBase:
    def __init__(self, name, logger_file_dir=None):
        self.logger_file_dir = logger_file_dir
        self.logger_file_path = None
        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"{name}.txt")

    @staticmethod
    def create_receiver(queue_receiver: mp.Queue, num_sender: int = -1):
        return Receiver(queue_receiver, num_sender)

    @staticmethod
    def create_tensor_receiver(queue_receiver, num_sender: int = -1, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return TensorReceiver(queue_receiver, num_sender=num_sender, device=device)


class MemoryMainBase(MainBase):
    def __init__(self, logger_file_dir=None):
        super().__init__("memory", logger_file_dir)

    def __call__(self, queue_sender: mp.Queue):
        """
        自此函数中实例化MemoryReplayServer对象，处理actor收集的数据.

        the train function will use this function like bellowing:

        import multiprocessing as mp
        queue_receiver = mp.Queue(maxsize=1)
        mp.Process(target=actor_server_main, args=(queue_receiver,), daemon=False, name="actor_server").start()

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

    def __call__(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
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
        self.main(queue_receiver, queue_sender)

    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        raise NotImplementedError


class LeagueMainBase(MainBase):
    def __init__(self, logger_file_dir=None):
        super().__init__("league", logger_file_dir)

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


def train_main(learner_main: LearnerMainBase,
               memory_mains: Union[List[MemoryMainBase], Tuple[MemoryMainBase]],
               league_main: LeagueMainBase,
               memory_buffer_length=1):  # receiver and sender

    mp.set_start_method("spawn")

    queue_receiver = mp.Queue(maxsize=memory_buffer_length)  # receiver batched tensor, when on policy, this can be set to 1
    queue_sender = mp.Queue(maxsize=1)  # the queue to send the newest data

    learner_process = mp.Process(target=learner_main, args=(queue_receiver, queue_sender),
                                 daemon=False, name="learner_main")
    learner_process.start()

    league_process = mp.Process(target=league_main, args=(queue_sender,),
                                daemon=False, name="league_main")
    league_process.start()

    memory_processes = []
    for i, memory_main in enumerate(memory_mains):
        memory_process = mp.Process(target=memory_main, args=(queue_receiver,),
                                           daemon=False, name=f"memory_main_{i}")
        memory_process.start()
        memory_processes.append(memory_process)

    try:
        learner_process.join()
        league_process.join()
        for memory_process in memory_processes:
            memory_process.join()
    finally:
        learner_process.close()
        league_process.close()
        for memory_process in memory_processes:
            memory_process.close()









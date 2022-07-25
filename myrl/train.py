import os
import multiprocessing as mp
import torch

import myrl.utils as utils
from myrl.memory_replay import TensorReceiver
from myrl.connection import Receiver
from myrl.actor import Actor, ActorClient


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
    def create_tensor_receiver(queue_receiver, num_sender=-1, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return TensorReceiver(queue_receiver, num_sender=num_sender, device=device)


class ActorCreateBase(MainBase):
    def __init__(self, logger_file_dir, steps):
        super().__init__(None)
        self.logger_file_dir = logger_file_dir
        self.steps = steps

    def __call__(self, actor_indexes: tuple, queue_gather2actor, queue_actor2gather):
        gather_id, actor_index, num_samples, num_evals = actor_indexes
        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"gather_{gather_id}_actor_{actor_index}.txt")
        utils.set_process_logger(file_path=self.logger_file_path)

        env, agent = self.create_env_and_agent()

        if actor_index < num_samples:
            actor = Actor(env, agent, steps=self.steps, get_full_episode=False)
            actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="sampler")
        else:
            actor = Actor(env, agent, steps=self.steps, get_full_episode=True)
            actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="evaluator")
        actor_client.run()

    def create_env_and_agent(self):
        raise NotImplementedError


class MemoryReplayMainBase(MainBase):
    def __init__(self, logger_file_dir=None):
        super().__init__("memory_replay", logger_file_dir)

    def __call__(self, queue_receiver: mp.Queue):
        """
        自此函数中实例化MemoryReplayServer对象，处理actor收集的数据.

        the train function will use this function like bellowing:

        import multiprocessing as mp
        queue_receiver = mp.Queue(maxsize=1)
        mp.Process(target=actor_server_main, args=(queue_receiver,), daemon=False, name="actor_server").start()

        :param queue_receiver: used to cache data generated
        :return: None
        """
        utils.set_process_logger(file_path=self.logger_file_path)
        self.main(queue_receiver)

    def main(self, queue_receiver: mp.Queue):
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
        mp.Process(target=learner_main, args=(queue_receiver,queue_sender), daemon=False, name="learner_main").start()

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

    def __call__(self, queue_sender: mp.Queue):
        """
           the process to manage model weights.

           the train function will use this function like bellowing:

           import multiprocessing as mp
           queue_send = mp.Queue(maxsize=1)
           mp.Process(target=learner_main, args=(queue_send), daemon=False, name="league_main").start()

           :param queue_sender: the queue to send model weights
           :return: None
        """
        utils.set_process_logger(file_path=self.logger_file_path)
        self.main(queue_sender)

    def main(self, queue_sender: mp.Queue):
        raise NotImplementedError


def train_main(learner_main: LearnerMainBase,
               memory_replay_main: MemoryReplayMainBase,
               league_main: LeagueMainBase,
               queue_size=(1, 1)):  # receiver and sender

    queue_receiver = mp.Queue(maxsize=queue_size[0])  # receiver batched tensor, when on policy, this can be set to 1
    queue_sender = mp.Queue(maxsize=queue_size[1])  # the queue to send the newest data

    learner_process = mp.Process(target=learner_main, args=(queue_receiver, queue_sender),
                                 daemon=False, name="learner_main")
    learner_process.start()

    league_process = mp.Process(target=league_main, args=(queue_sender,),
                                daemon=False, name="league_main")
    league_process.start()

    memory_replay_main = mp.Process(target=memory_replay_main, args=(queue_receiver,),
                                    daemon=False, name="actor_server_main")
    memory_replay_main.start()

    try:
        learner_process.join()
        league_process.join()
        memory_replay_main.join()
    finally:
        learner_process.close()
        league_process.close()
        memory_replay_main.close()









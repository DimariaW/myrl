
import logging
import time
from myrl.memory_replay import MultiProcessTrajQueueServer, Receiver
import torch
import numpy as np
import multiprocessing as mp
from myrl.utils import set_process_logger
import threading


class Learner:
    def __init__(self, mr):
        self.mr = mr

    def run(self):
        while True:
            data = self.mr.recall()
            time.sleep(1)
            logging.info(list(data.keys()))
            logging.info(data["action"].shape)


def run_learn(batch_client):
    set_process_logger(file_path=f"./log/test_mp_traj_queue_server_client/learner.txt",
                       file_level=logging.DEBUG, starts_with=None)
    learner = Learner(batch_client)
    learner.run()


def test():
    mp.set_start_method("spawn")
    logname = "test_mp_traj_queue_server_client"
    set_process_logger(file_path=f"./log/{logname}/main.txt",
                       file_level=logging.DEBUG, starts_with=None)

    queue_receiver = mp.Queue(maxsize=8)
    batch_server = MultiProcessTrajQueueServer(maxlen=8, queue_receiver=queue_receiver,
                                               batch_size=8, num_batch_maker=4,
                                               logger_file_path=f'./log/{logname}/batch_maker.txt')
    batch_server.start()

    batch_client = Receiver(queue_receiver)
    mp.Process(target=run_learn, args=(batch_client,), daemon=True, name="learner").start()
    while True:
        episode = [{"obs": {"state": np.random.rand(53, 11, 11), "mask": np.random.rand(8)},
                    "action": np.random.randint(8),
                    "done": False
                    } for _ in range(4 )
                   ]
        batch_server.cache(episode)

import logging
import queue
import time
import torch
import numpy as np
import multiprocessing as mp

import myrl.train as core
import myrl.memory_replay as mr


class MemoryReplayMain(core.MemoryReplayMainBase):
    def main(self, queue_receiver: mp.Queue):
        #traj_list = mr.TrajList(queue_receiver)
        traj_queue = mr.TrajQueueMP(maxlen=16, queue_receiver=queue_receiver, batch_size=16,
                                    num_batch_maker=3, logger_file_dir="./log/test_traj_queue_mp/")
        traj_queue.start()
        for _ in range(64):
            for _ in range(16):
                episode = [{f"key{i}": np.random.randn(11, 11) for i in range(20)
                            } for _ in range(64)
                           ]
                #traj_list.cache(episode)
                traj_queue.cache(episode)
        traj_queue.stop()
        time.sleep(2)
            #traj_list.start()
        #traj_list.stop()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        device = torch.device("cpu")
        tensor_receiver = self.create_tensor_receiver(queue_receiver, num_sender=3, device=device)
        num = 0
        try:
            while True:
                tensor = tensor_receiver.recall()
                num += 1
                logging.debug(f"successfully receive {num}")
                time.sleep(0.01)
                queue_sender.put((False, tensor))
                logging.debug(f"successfully put {num}")
        except queue.Empty:
            logging.debug("end receiving")
            queue_sender.put((True, None))


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_sender: mp.Queue):
        receiver = self.create_tensor_receiver(queue_sender, num_sender=1)
        num = 0
        try:
            while True:
                data = receiver.recall()
                num += 1
                logging.info(f"receive num {num}")
                logging.info(list(data.keys()))
        except queue.Empty:
            logging.info("end receiving")

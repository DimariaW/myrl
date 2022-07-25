import logging
import queue
import time
import torch
import numpy as np
import multiprocessing as mp

import myrl.train as core


class MemoryReplayMain(core.MemoryReplayMainBase):
    def main(self, queue_receiver: mp.Queue):
        num = 0
        for i in range(100):
            queue_receiver.put((False, np.random.randn(3, 512, 512)))
            num += 1
            logging.info(f"put num {num}")
        queue_receiver.put((True, None))


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        device = torch.device("cpu")
        tensor_receiver = self.create_tensor_receiver(queue_receiver, num_sender=1, device=device)
        num = 0
        try:
            while True:
                tensor = tensor_receiver.recall()
                num += 1
                logging.info(f"successfully receive {num}")
                time.sleep(1)
                queue_sender.put((False, torch.mean(tensor)))
                logging.info(f"successfully put {num}")
        except queue.Empty:
            logging.info("end receiving")
            queue_sender.put((True, None))


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_sender: mp.Queue):
        receiver = self.create_tensor_receiver(queue_sender, num_sender=1)
        num = 0
        try:
            while True:
                num += 1
                logging.info(f"receive num {num} and value {repr(receiver.recv())}")
        except queue.Empty:
            logging.info("end receiving")






import logging

from myrl.memory_replay import MultiProcessBatcher
import torch
import numpy as np
import multiprocessing as mp
from myrl.utils import set_process_logger

if __name__ == "__main__":
    mp.set_start_method("spawn")

    set_process_logger(file_path="./log/test_mp_batcher/main.txt",
                       file_level=logging.DEBUG)

    batcher = MultiProcessBatcher(maxlen=3000, device=torch.device("cpu"),
                                  batch_size=192, forward_steps=64, num_batch_maker=4, use_queue=True,
                                  logger_file_path="./log/test_mp_batcher/test.txt",
                                  file_level=logging.DEBUG)

    episodes = []
    for i in range(300):
        episode = [{"obs": {"state": np.random.rand(10), "mask": np.random.rand(8)},
                    "action": np.random.randint(8),
                    "done": False
                    } for _ in range(32)
                   ]
        episodes.append(episode)

    batcher.cache(episodes)

    batcher.start()

    while True:
        data = batcher.recall()
        print(list(data.keys()))
        print(data["done"].shape)





import logging
import time
from myrl.memory_replay import MultiProcessBatcher
import torch
import numpy as np
import multiprocessing as mp
from myrl.utils import set_process_logger
import threading

def consume(bathcer):
    while True:
        if len(bathcer) > 400:
            break
    batcher.start()
    while True:
        time.sleep(1)
        data = batcher.recall()
        print(list(data.keys()))
        print(data["done"].shape)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    set_process_logger(file_path="./log/test_mp_batcher/main.txt",
                       file_level=logging.DEBUG, starts_with=None)

    batcher = MultiProcessBatcher(maxlen=3000, device=torch.device("cpu"),
                                  batch_size=192, forward_steps=32, num_batch_maker=4, use_queue=True,
                                  logger_file_path="./log/test_mp_batcher/test.txt",
                                  file_level=logging.DEBUG)

    threading.Thread(target=consume, args=(batcher,), daemon=True).start()
    while True:
        episode = [{"obs": {"state": np.random.rand(10), "mask": np.random.rand(8)},
                    "action": np.random.randint(8),
                    "done": False
                    } for _ in range(64)
                   ]
        batcher.cache(episode)







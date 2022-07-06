import logging
from myrl.memory_replay import MultiProcessBatcher
import myrl.connection as connection
import numpy as np
import time
import myrl.utils as utils


def func(data):
    return data


class SendGenerator:
    def __iter__(self):
        return self

    def __next__(self):
        return np.random.rand(64, 128)




def test_multiprocess_job_executor():
    utils.set_process_logger(file_path="./log/test.txt")
    worker = connection.MultiProcessJobExecutors(func, SendGenerator(), 4,
                                                 name_prefix="batch data process",
                                                 logger_file_path="./log/batch_data_process.txt",
                                                 num_receivers=2)
    worker.start()
    while True:
        if worker.output_queue.not_empty:
            logging.info(worker.output_queue.get().shape)


def test_multiprocess_batcher():
    batcher = MultiProcessBatcher(maxlen=1000, batch_size=192, forward_steps=64, num_batch_maker=2)
    utils.set_process_logger(file_path="./log/test2.txt")
    for _ in range(200):
        episode = []
        for _ in range(100):
            moment = dict()
            moment["observation"] = np.random.randn(10)
            moment["action"] = 0
            moment["behavior_log_prob"] = -1
            moment["reward"] = 0
            moment["done"] = False
            episode.append(moment)
        batcher.cache(episode)

    batcher.start()

    time.sleep(10)
    while True:
        data = batcher.recall()
        data_info = {}
        for key, value in data.items():
            data_info[key] = value.shape
        logging.info(data_info)

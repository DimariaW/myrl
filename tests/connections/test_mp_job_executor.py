import logging
import myrl.connection as connection
import numpy as np

import myrl.utils as utils
import multiprocessing as mp
import time


def func(data):
    logging.debug("start process data")
    logging.debug("finish process data")
    return data


def send_generator():
    while True:
        yield np.random.rand(32, 64, 128)


def test_multiprocess_job_executor_v2():
    mp.set_start_method("spawn")
    utils.set_process_logger(file_path="./log/test_mp_job_executor_v2/main.txt")
    worker = connection.MultiProcessJobExecutorsV2(func, send_generator(), num=16,
                                                 #num_receivers=4,
                                                   buffer_length=8,
                                                   name_prefix="test",
                                                   logger_file_path="./log/test_mp_job_executor_v2/test.txt",
                                                   file_level=logging.DEBUG)
    worker.start()
    while True:
        if worker.output_queue.not_empty:
            #time.sleep(1)
            logging.info(worker.recv().shape)


def test_multiprocess_job_executor():
    mp.set_start_method("spawn")
    utils.set_process_logger(file_path="./log/test_mp_job_executor/main.txt")
    worker = connection.MultiProcessJobExecutors(func, send_generator(), num=4,
                                                 num_receivers=1,
                                                 buffer_length=8,
                                                 name_prefix="test",
                                                 logger_file_path="./log/test_mp_job_executor/test.txt",
                                                 file_level=logging.DEBUG)
    worker.start()
    while True:
        if worker.output_queue.not_empty:
            #time.sleep(1)
            logging.info(worker.recv().shape)

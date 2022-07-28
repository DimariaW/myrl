import logging
import queue

import myrl.connection as connection
import myrl.utils as utils
import multiprocessing as mp
import time


def func(data):
    logging.debug("start process data")
    i = 0
    for _ in range(10000000):
        i += 1
    logging.debug("finish process data")
    return data


def send_generator():
    for i in range(30):
    #while True:
        yield i

#%%


def test_multiprocess_job_executor():
    mp.set_start_method("spawn")
    log_dir = "test_multiprocess_job_executor"
    utils.set_process_logger(file_path=f"./log/{log_dir}/main.txt")
    worker = connection.MultiProcessJobExecutors(func, send_generator(), num=4,
                                                 buffer_length=3,
                                                 name_prefix="worker",
                                                 logger_file_dir=f"./log/{log_dir}/")
    beg = time.time()
    worker.start()
    while True:
        try:
            time.sleep(0.1)
            logging.info(worker.recv())
        except queue.Empty:
            end = time.time()
            logging.info(f"mp time consume {end-beg}")
            break

    beg = time.time()
    for data in send_generator():
        logging.info(func(data))
    end = time.time()
    logging.info(f"single process consume {end-beg}")

#%%


def receiver_process(receiver: connection.Receiver, logger_file_path=None):
    utils.set_process_logger(file_path=logger_file_path)
    while True:
        try:
            time.sleep(1)
            logging.info(receiver.recv())
        except queue.Empty:
            logging.info("successfully receive all data")
            break

    for data in send_generator():
        logging.info(func(data))


def test_multiprocess_job_executor_with_receiver():
    #mp.set_start_method("spawn")
    log_dir = "test_multiprocess_job_executor_with_receiver"
    utils.set_process_logger(file_path=f"./log/{log_dir}/main.txt")
    queue_receiver = mp.Queue(maxsize=1)
    worker = connection.MultiProcessJobExecutors(func, send_generator(), num=4,
                                                 buffer_length=3,
                                                 queue_receiver=queue_receiver,
                                                 name_prefix="work",
                                                 logger_file_dir=f"./log/{log_dir}/",
                                                 file_level=logging.DEBUG)

    receiver = connection.Receiver(queue_receiver, num_sender=4)
    mp.Process(target=receiver_process, name="receiver", args=(receiver,
                                                               f"./log/{log_dir}/receiver.txt"), daemon=True).start()
    worker.start()


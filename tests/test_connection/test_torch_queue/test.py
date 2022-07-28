import queue
import multiprocessing as mp
import numpy as np
import time
import torch
from myrl.connection import Receiver, MultiProcessJobExecutors
from myrl.memory import ToTensorWrapper
from myrl.utils import set_process_logger
import logging
from tests.test_connection.test_torch_queue.fun import process, test_process


def send_generator():
    for _ in range(20):
        yield np.random.randn(1024, 1024, 52)


process = ToTensorWrapper(device=torch.device("cpu"), func=process)


if __name__ == "__main__":
    set_process_logger()
    receiver = torch.multiprocessing.Queue(maxsize=1)
    receiver = mp.Queue(maxsize=4)
    mp.Process(target=test_process, args=(receiver,)).start()
    beg = time.time()
    for _ in range(20):

        logging.info(receiver.get().shape)


    logging.info(f"time.consume: {time.time() - beg}")






    """
    set_process_logger()
    receiver = torch.multiprocessing.Queue(maxsize=4)
    #receiver = mp.Queue(maxsize=4)
    worker = MultiProcessJobExecutors(func=process, send_generator=send_generator(), num=4, buffer_length=4,
                                      queue_receiver=receiver)

    beg = time.time()
    receiver = Receiver(receiver, num_sender=4)
    worker.start()

    while True:
        try:
            logging.info(receiver.recv().shape)
        except queue.Empty:
            logging.info(f"time.consume: {time.time() - beg}")
            break
    """
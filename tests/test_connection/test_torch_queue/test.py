import queue
import multiprocessing as mp
import numpy as np
import time
import torch
from myrl.connection import Receiver, MultiProcessJobExecutors
from myrl.memory import ToTensorWrapper
from myrl.utils import set_process_logger, to_tensor
import logging
from tests.test_connection.test_torch_queue.fun import process, test_process

import functools


def send_generator():
    for _ in range(20):
        yield np.random.randn(1024, 1024, 52)


if __name__ == "__main__":
    """
    set_process_logger()
    #receiver = torch.multiprocessing.Queue(maxsize=1)
    receiver = mp.Queue(maxsize=1)
    mp.Process(target=test_process, args=(receiver,)).start()
    beg = time.time()
    for _ in range(20):

        logging.info(receiver.get().shape)


    logging.info(f"time.consume: {time.time() - beg}")
    """





    mp.set_start_method("spawn")
    set_process_logger()
    #receiver = torch.multiprocessing.Queue(maxsize=4)
    queue_receiver = mp.Queue(maxsize=4)
    worker = MultiProcessJobExecutors(func=process, send_generator=send_generator(), num=4, buffer_length=4,
                                      queue_receiver=queue_receiver, waiting_time=10)

    beg = time.time()
    receiver = Receiver(queue_receiver, num_sender=4, postprocess=functools.partial(to_tensor,
                                                                                    device=torch.device("cpu"),
                                                                                    unsqueeze=None))
    worker.start()

    while True:
        try:
            logging.info(receiver.recv().shape)
        except queue.Empty:
            logging.info(f"time.consume: {time.time() - beg}")
            break

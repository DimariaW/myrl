import myrl.connection as connection

import threading
import multiprocessing as mp
import numpy as np
import os.path as osp
import logging


class League:
    def __init__(self, port: int, actor_num: int = None,
                 queue_receiver: connection.Receiver = None,
                 model_weights_save_dir: str = None,
                 model_weights_save_intervals: int = None
                 ):
        self.port = port
        self.actor_num = actor_num
        self.actor_communicator = connection.QueueCommunicator(port, num_client=self.actor_num)

        self.queue_receiver = queue_receiver

        self.cached_weights = None
        self.num_model_weights_update = 0
        self.old_num_model_weights_update = 0

        self.model_weights_save_dir = model_weights_save_dir
        self.model_weights_save_intervals = model_weights_save_intervals

    def _receive(self):
        while True:
            self._receive_once()

    def _receive_once(self):
        self.num_model_weights_update, self.cached_weights = self.queue_receiver.recv()

        logging.debug(f"weights update times is {self.num_model_weights_update}")
        if self.num_model_weights_update - self.old_num_model_weights_update > self.model_weights_save_intervals:
            np.save(file=osp.join(self.model_weights_save_dir, f"model_{self.num_model_weights_update}.npy"))
            self.old_num_model_weights_update = self.num_model_weights_update

    def run(self):
        """
        1. 等待learner 从queue_receiver 中传模型权重， 传save_intervals 步之后保存权重文件
        2. 向actor传模型权重
        """
        threading.Thread(target=self._receive, args=(), daemon=True).start()

        while True:
            if self.cached_weights is not None:
                break

        self.actor_communicator.run()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            logging.debug(cmd)
            assert(cmd == "model")
            self.actor_communicator.send(conn, (cmd, self.cached_weights))

    def run_sync(self):
        self.actor_communicator.run_sync()

        conns = []
        self._receive_once()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()

            logging.debug(cmd)
            assert (cmd == "model")

            conns.append(conn)
            if len(conns) == self.actor_num:
                for conn in conns:
                    self.actor_communicator.send(conn, (cmd, self.cached_weights))

                conns = []
                self._receive_once()







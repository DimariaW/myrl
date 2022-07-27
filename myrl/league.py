import myrl.connection as connection
import myrl.utils as utils

import tensorboardX
import threading
import numpy as np
import os
import os.path as osp
import logging
import bz2
import pickle

from typing import Dict
from collections import defaultdict


class League:
    def __init__(self,
                 queue_receiver: connection.Receiver,

                 port: int,
                 num_actors: int = None,

                 model_weights_save_dir: str = None,
                 model_weights_save_intervals: int = None,

                 tensorboard_dir: str = None,

                 use_bz2: bool = True
                 ):
        """
        1. 从learner端接收模型队列
        """
        self.queue_receiver = queue_receiver
        """
        2. 与actor通信模块
        """
        self.port = port
        self.num_actors = num_actors
        self.actor_communicator = connection.QueueCommunicator(port, num_client=self.num_actors)
        """
        3. cached_weights and saved_weights
        """
        self.cached_weights = None
        self.num_cached_weights_update = -1
        self.saved_weights = None
        self.num_saved_weights_update = -1

        if model_weights_save_dir is not None:
            self.model_weights_save_dir = model_weights_save_dir
            os.makedirs(self.model_weights_save_dir, exist_ok=True)
            self.model_weights_save_intervals = model_weights_save_intervals
        """
        4. use tensorboard to log eval infos
        """
        if tensorboard_dir is not None:
            self.num_received_eval_infos = 0
            self.sw = tensorboardX.SummaryWriter(logdir=tensorboard_dir)

        self.use_bz2 = use_bz2

    @utils.wrap_traceback
    def _receive(self):
        while True:
            self._receive_once()

    def _receive_once(self):
        """
        1. 更新cached weights
        """
        self.num_cached_weights_update, weights = self.queue_receiver.recv()
        if not self.use_bz2:
            self.cached_weights = weights
        else:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        """
        2. 在开始阶段更新saved_weights
        """
        if self.num_saved_weights_update < 0:
            self.saved_weights = self.cached_weights
            self.num_saved_weights_update = self.num_cached_weights_update

        logging.debug(f"weights update times is {self.num_cached_weights_update}")
        """
        3. 如果需要save model weights, 则只在固定时间步之后更新save, or update saved_weights to cached_weights
        """
        if hasattr(self, "model_weights_save_dir"):
            if self.num_cached_weights_update - self.num_saved_weights_update >= self.model_weights_save_intervals:
                np.save(file=osp.join(self.model_weights_save_dir, f"model_{self.num_cached_weights_update}.npy"),
                        arr=weights)

                self.saved_weights = self.cached_weights
                self.num_saved_weights_update = self.num_cached_weights_update
        else:
            self.saved_weights = self.cached_weights
            self.num_saved_weights_update = self.num_cached_weights_update

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
            """
            cmd: Literal["model", "eval_infos"]
            """
            if cmd == "model":
                logging.debug(f"cmd: {cmd}, data: {data}")
                if data == "latest":
                    self.actor_communicator.send(conn,
                                                 (cmd, (self.num_cached_weights_update, self.cached_weights)))
                elif data == -1:
                    self.actor_communicator.send(conn,
                                                 (cmd, (self.num_saved_weights_update,
                                                        self.saved_weights)))

            elif cmd == "eval_infos":
                logging.debug(f"cmd:{cmd}")
                self._record_eval_info(data)
                self.actor_communicator.send(conn, (cmd, "successfully sent eval info"))

    def _record_eval_info(self, data: Dict[str, list]):
        if hasattr(self, "sw"):
            num_received = 0
            for key, values in data.items():
                num_received = len(values)
                for i, value in enumerate(values):
                    self.sw.add_scalar(tag=key, scalar_value=value, global_step=self.num_received_eval_infos + i)
            self.num_received_eval_infos += num_received
        else:
            logging.debug(data)

    def run_sync(self):
        self.actor_communicator.run_sync()

        conns = []
        self._receive_once()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()

            logging.debug(cmd)

            if cmd == "model":
                conns.append(conn)
                if len(conns) == self.actor_num:
                    for conn in conns:
                        self.actor_communicator.send(conn, (cmd, (self.num_model_weights_update, self.cached_weights)))
                    conns = []
                    self._receive_once()

            if cmd == "eval_info":
                self._record_eval_info(data)
                self.actor_communicator.send(conn, (cmd, "successfully record eval info"))








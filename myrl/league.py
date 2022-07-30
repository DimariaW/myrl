import myrl.connection as connection
import myrl.utils as utils
import myrl.model as model

import tensorboardX
import threading
import numpy as np
import os
import os.path as osp
import logging
import bz2
import pickle
import torch

from typing import Dict
from collections import defaultdict


class League:
    def __init__(self,
                 queue_receiver: connection.Receiver,

                 port: int,
                 num_actors: int = None,

                 model_weights_cache_intervals: int = None,
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
        self.num_update, self.model = self.queue_receiver.recv()
        self.num_update: torch.Tensor
        self.model: model.Model

        self.use_bz2 = use_bz2

        weights = self.model.get_weights()
        if self.use_bz2:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        else:
            self.cached_weights = weights
        self.num_cached_weights_update = self.num_update.item()
        self.model_weights_cache_intervals = model_weights_cache_intervals

        if model_weights_save_dir is not None:
            self.model_weights_save_dir = model_weights_save_dir
            os.makedirs(self.model_weights_save_dir, exist_ok=True)
            self.model_weights_save_intervals = model_weights_save_intervals

        self.saved_weights = self.cached_weights
        self.num_saved_weights_update = self.num_cached_weights_update

        """
        4. use tensorboard to log eval infos
        """
        if tensorboard_dir is not None:
            self.num_received_eval_infos = defaultdict(lambda: 0)
            self.sw = tensorboardX.SummaryWriter(logdir=tensorboard_dir)

    @utils.wrap_traceback
    def _update(self):
        while True:
            self._update_once()

    def _update_once(self):
        """
        1. 更新cached weights
        """
        if self.num_update.item() - self.num_cached_weights_update >= self.model_weights_cache_intervals:
            self._update_cached_weights()
            """ 
            2. 如果需要save model weights, 则只在固定时间步之后更新save, or update saved_weights to cached_weights
            """
            self._update_saved_weights()

    def _update_cached_weights(self):
        self.num_cached_weights_update = self.num_update.item()
        weights = self.model.get_weights()
        if not self.use_bz2:
            self.cached_weights = weights
        else:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        logging.debug(f"successfully update model to model id {self.num_cached_weights_update}")

    def _update_saved_weights(self):
        if hasattr(self, "model_weights_save_dir"):
            if self.num_cached_weights_update - self.num_saved_weights_update >= self.model_weights_save_intervals:
                np.save(file=osp.join(self.model_weights_save_dir, f"model_{self.num_cached_weights_update}.npy"),
                        arr=self.cached_weights)

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
        threading.Thread(target=self._update, args=(), daemon=True).start()

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
            for key, values in data.items():
                for value in values:
                    self.num_received_eval_infos[key] += 1
                    self.sw.add_scalar(tag=key, scalar_value=value, global_step=self.num_received_eval_infos[key])
        else:
            logging.debug(data)

    def run_sync(self):
        self.actor_communicator.run_sync()

        conns = []

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()

            logging.debug(cmd)

            if cmd == "model":
                conns.append(conn)
                if len(conns) == self.num_actors:
                    for conn in conns:
                        self.actor_communicator.send(conn, (cmd, (self.num_cached_weights_update, self.cached_weights)))
                    conns = []
                    while self.num_update.item() <= self.num_cached_weights_update:
                        pass
                    self._update_cached_weights()
                    self._update_saved_weights()

            if cmd == "eval_info":
                self._record_eval_info(data)
                self.actor_communicator.send(conn, (cmd, "successfully record eval info"))








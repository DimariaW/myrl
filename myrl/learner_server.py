import myrl.connection as connection
import threading
from myrl.algorithm import Algorithm
import numpy as np
import logging
from tensorboardX import SummaryWriter


class ActorCommunicator(connection.QueueCommunicator):
    def __init__(self, port: int, actor_num=None):
        super().__init__()
        self.port = port
        self.actor_num = actor_num

    def run(self):

        def worker_server(port):
            logging.info('started actor server %d' % port)
            conn_acceptor = connection.accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)

                self.add_connection(conn)

                logging.info(f"total actor count now is {self.connection_count()}")

        threading.Thread(name="actor server", target=worker_server, args=(self.port,), daemon=True).start()

    def run_sync(self):
        """
        同步，堵塞直到所有actor建立连接
        """
        if self.actor_num is None:
            raise ValueError("sync version requires known actor num")

        logging.info('started actor server %d' % self.port)
        conn_acceptor = connection.accept_socket_connections(port=self.port, maxsize=self.actor_num)
        while True:
            try:
                conn = next(conn_acceptor)
                self.add_connection(conn)
                logging.info(f"total actor count now is {self.connection_count()}")
            except StopIteration:
                break


class LearnerServer:
    def __init__(self, learner: Algorithm = None, port: int = None,
                 tensorboard_log_dir=None, actor_num=None, sampler_num=None):
        self.actor_communicator = ActorCommunicator(port, actor_num)
        self.learner = learner
        self.sampler_num = sampler_num

        self.cached_weights = None

        self.sw = SummaryWriter(logdir=tensorboard_log_dir)
        self.sample_reward_steps = 0
        self.eval_reward_steps = 0

    def run(self):
        logging.info("start running learner server")
        self.cached_weights = self.learner.get_weights()
        last_update_num_cached = 0

        self.actor_communicator.run()
        threading.Thread(name="learn thread", target=self.learner.run, args=(), daemon=True).start()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            logging.debug(f"received cmd : {cmd}")

            if cmd == "model":
                self.actor_communicator.send(conn, (cmd, self.learner.get_weights()))
                #self.actor_communicator.send(conn, (cmd, self.cached_weights))

            elif cmd == "episode":
                """
                for i, moment in enumerate(data):
                    if i < len(data) - 1:
                        self.learner.memory_replay.cache(moment["observation"], moment["action"],
                                                         data[i + 1]["observation"], moment["reward"], moment["done"])
                    elif moment["done"] is True:
                        self.learner.memory_replay.cache(moment["observation"], moment["action"],
                                                         np.random.rand(*moment["observation"].shape),
                                                         moment["reward"], moment["done"])
                """
                self.learner.memory_replay.cache(data)
                self.actor_communicator.send(conn, (cmd, "successfully sent episodes"))

                #if (self.learner.memory_replay.num_cached - last_update_num_cached) >= 400:
                 #   logging.info("update cached weights")
                  #  self.cached_weights = self.learner.get_weights()
                   # last_update_num_cached = self.learner.memory_replay.num_cached

            elif cmd == "sample_reward":
                for reward in data:
                    self.sample_reward_steps += 1
                    self.sw.add_scalar(tag="sample_reward", scalar_value=reward, global_step=self.sample_reward_steps)
                self.actor_communicator.send(conn, (cmd, "successfully sent sample rewards"))

            elif cmd == "eval_reward":
                for reward in data:
                    self.eval_reward_steps += 1
                    self.sw.add_scalar(tag="eval_reward", scalar_value=reward, global_step=self.eval_reward_steps)
                self.actor_communicator.send(conn, (cmd, "successfully sent eval rewards"))

    def run_on_policy(self):
        self.actor_communicator.run_sync()

        received_episodes_num = 0
        conns = []
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            if cmd == "model":
                self.actor_communicator.send(conn, (cmd, self.learner.get_weights()))
            elif cmd == "episodes":
                self.learner.memory_replay.cache(data)
                received_episodes_num += 1
                conns.append(conn)

                if received_episodes_num == self.sampler_num:
                    logging.info(self.learner.learn())

                    for conn in conns:
                        self.actor_communicator.send(conn, (cmd, "successfully received episodes"))

                    received_episodes_num = 0
                    conns = []

    def run_test(self):
        """
        function to test actor
        :return:
        """
        self.actor_communicator.run()
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            logging.info((cmd, data))
            self.actor_communicator.send(conn, (cmd, "successfully receive"))




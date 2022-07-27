
import torch
import multiprocessing as mp
import os
import myrl.core as core
import myrl.memory as mem
import myrl.algorithm as alg
import myrl.league as lg
import numpy as np

from tests.football.football_model import CNNModel

from tests.football.config import USE_BZ2


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        traj_queue = mem.TrajQueueMP(maxlen=32,
                                     queue_sender=queue_sender,
                                     batch_size=16,
                                     use_bz2=USE_BZ2,
                                     num_batch_maker=4,
                                     logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"))

        memory_server = mem.MemoryServer(traj_queue, self.port, actor_num=None,
                                         tensorboard_dir=os.path.join(self.logger_file_dir, "sample_reward"))
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        tensor_receiver = self.create_tensor_receiver(queue_receiver, device=device)
        model = CNNModel((16, 72, 96), 19).to(device)
        model.set_weights(np.load("./tests/football/easy_model/model_346346.npy", allow_pickle=True).item())
        impala = alg.IMPALA(model, tensor_receiver,
                            lr=0.00019896, gamma=0.993, lbd=1, vf=0.5, ef=0.00087453,
                            queue_sender=queue_sender,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            use_upgo=False,
                            send_intervals=1)
        impala.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        league = lg.League(queue_receiver, self.port,
                           model_weights_save_dir=os.path.join(self.logger_file_dir, "model"),
                           model_weights_save_intervals=1000,
                           tensorboard_dir=os.path.join(self.logger_file_dir, "eval_info"),
                           use_bz2=USE_BZ2)
        league.run()

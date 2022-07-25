import torch
import multiprocessing as mp

import myrl.train as core
import myrl.memory_replay as mr
import myrl.algorithm as alg
import myrl.league as lg

from tests.a2c_rebuild.model import Model


class MemoryReplay(core.MemoryReplayMainBase):
    def main(self, queue_receiver: mp.Queue):
        traj_list = mr.TrajList(queue_receiver)
        memory_server = mr.MemoryReplayServer(traj_list, 7777, actor_num=10, tensorboard_dir=self.logger_file_dir)
        memory_server.run_sync()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        memory_replay = self.create_tensor_receiver(queue_receiver, device=device)
        model = Model(8, 4, use_orthogonal_init=True, use_tanh=True)
        a2c = alg.A2C(model, memory_replay,
                      lr=2e-3, gamma=0.99, lbd=0.98, vf=0.5, ef=1e-2,
                      queue_sender=queue_sender, tensorboard_dir=self.logger_file_dir)
        a2c.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_sender: mp.Queue):
        league = lg.League(7778, actor_num=10, queue_sender=queue_sender, save_dir=self.logger_file_dir, save_intervals=10)
        league.run()

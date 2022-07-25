import torch
import multiprocessing as mp
import os
import myrl.train as core
import myrl.memory_replay as mr
import myrl.algorithm as alg
import myrl.league as lg

from tests.impala_rebuild.model import Model


class MemoryReplayMain(core.MemoryReplayMainBase):
    def main(self, queue_receiver: mp.Queue):
        traj_queue = mr.TrajQueue(maxlen=8, queue_receiver=queue_receiver, batch_size=4)
        memory_server = mr.MemoryReplayServer(traj_queue, 7777, actor_num=None,
                                              tensorboard_dir=os.path.join(self.logger_file_dir, "reward"))
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        memory_replay = self.create_tensor_receiver(queue_receiver, device=device)
        model = Model(8, 4, use_orthogonal_init=True, use_tanh=True)
        impala = alg.IMPALA(model, memory_replay,
                            lr=2e-3, gamma=0.99, lbd=0.98, vf=0.5, ef=1e-2,
                            queue_sender=queue_sender,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            use_upgo=True, send_intervals=1)
        impala.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_sender: mp.Queue):
        queue_receiver = self.create_receiver(queue_sender)
        league = lg.League(7778, actor_num=None, queue_receiver=queue_receiver,
                           model_weights_save_dir=os.path.join(self.logger_file_dir, "model"),
                           model_weights_save_intervals=1000)
        league.run()

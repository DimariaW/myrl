import torch
import multiprocessing as mp
import os
import myrl.core as core

import myrl.memory as mem
import myrl.algorithm as alg
import myrl.league as lg

from tests.impala_rebuild.model import Model


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        traj_queue = mem.TrajQueueMP(maxlen=8, queue_sender=queue_sender, batch_size=16,
                                     use_bz2=False, num_batch_maker=1,
                                     logger_file_dir=os.path.join(self.logger_file_dir, "batcher"))
        memory_server = mem.MemoryServer(traj_queue, self.port, actor_num=None,
                                         tensorboard_dir=os.path.join(self.logger_file_dir, "sample_reward"))
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor_receiver = self.create_receiver(queue_receiver, device=device)
        model = Model(8, 4, use_orthogonal_init=True, use_tanh=False).to(device)
        impala = alg.IMPALA(model, tensor_receiver,
                            lr=2e-4, gamma=0.99, lbd=0.98, vf=0.5, ef=1e-4,
                            queue_senders=queue_senders,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            upgo_key=None)
        impala.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        league = lg.League(queue_receiver=queue_receiver,
                           port=self.port, num_actors=None,
                           model_weights_cache_intervals=1,
                           model_weights_save_dir=os.path.join(self.logger_file_dir, "model"),
                           model_weights_save_intervals=1000,
                           use_bz2=False,
                           tensorboard_dir=os.path.join(self.logger_file_dir, "eval_info"))
        league.run()

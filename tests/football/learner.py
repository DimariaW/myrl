
import torch
import multiprocessing as mp
import os
import myrl.train as core
import myrl.memory_replay as mr
import myrl.algorithm as alg
import myrl.league as lg

from tests.football.football_model import CNNModel


class MemoryReplayMain(core.MemoryReplayMainBase):
    def main(self, queue_receiver: mp.Queue):
        traj_queue = mr.TrajQueueMP(maxlen=32,
                                    queue_receiver=queue_receiver,
                                    batch_size=16,
                                    num_batch_maker=4,
                                    logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"))
        memory_server = mr.MemoryReplayServer(traj_queue, 7777, actor_num=None,
                                              tensorboard_dir=os.path.join(self.logger_file_dir, "reward"))
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_sender: mp.Queue):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        memory_replay = self.create_tensor_receiver(queue_receiver, device=device)
        model = CNNModel((16, 72, 96), 19).to(device)
        impala = alg.IMPALA(model, memory_replay,
                            lr=0.00019896, gamma=0.993, lbd=1, vf=0.5, ef=0.00087453,
                            queue_sender=queue_sender,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            use_upgo=False,
                            send_intervals=1)
        impala.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_sender: mp.Queue):
        queue_receiver = self.create_receiver(queue_sender)
        league = lg.League(7778, actor_num=None, queue_receiver=queue_receiver,
                           model_weights_save_dir=os.path.join(self.logger_file_dir, "model"),
                           model_weights_save_intervals=1000)
        league.run()

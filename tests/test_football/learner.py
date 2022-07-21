
import multiprocessing as mp
from envs.football.football_model import FootballNet, SimpleModel
from myrl import LearnerServer
from myrl.utils import set_process_logger
from myrl.algorithm import IMPALA
from myrl.memory_replay import MultiProcessBatcher, MultiProcessTrajQueue

import torch

if __name__ == "__main__":
    mp.set_start_method("spawn")
    set_process_logger(file_path="./log/11_vs_11_easy_stochastic/learner.txt")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = FootballNet().to(device)
    #model = SimpleModel(2, 1).to(device)
    #mr = MultiProcessBatcher(maxlen=30000, device=device, batch_size=192, forward_steps=64, num_batch_maker=2,
                            # use_queue=True, logger_file_path="./log/empty_goal/batcher.txt")
    mr = MultiProcessTrajQueue(maxlen=8, device=device, batch_size=128,
                               num_batch_maker=2,
                               logger_file_path="./log/11_vs_11_easy_stochastic/batcher.txt")
    learner = IMPALA(model, mr, lr=1e-3, ef=1e-3, vf=0.5, gamma=0.993, lbd=1, upgo=False)
    learner_server = LearnerServer(learner, port=58899, tensorboard_log_dir="./log/11_vs_11_easy_stochastic/tensorboard/")
    learner_server.run()


from envs.football.football_model import SimpleModel
from myrl import LearnerServer
from myrl.utils import set_process_logger
from myrl.algorithm import IMPALA
from myrl.memory_replay import MultiProcessBatcher

import torch

if __name__ == "__main__":
    set_process_logger(file_path="./log/learner.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(2, 1).to(device)
    mr = MultiProcessBatcher(maxlen=3000, device=device, batch_size=64, forward_steps=32, num_batch_maker=2)
    learner = IMPALA(model, mr, lr=1e-3, ef=1e-3, vf=0.5, gamma=0.993, lbd=1.)
    learner_server = LearnerServer(learner, port=8010)
    learner_server.run()

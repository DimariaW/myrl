from myrl import ActorServer
from myrl.utils import set_process_logger
from myrl.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from myrl.algorithm import IMPALA
from myrl.memory import MultiProcessBatcher


class DuelNet(Model):
    def __init__(self, obs_dim, num_acts):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, num_acts + 1)

    def forward(self, obs):
        obs = list(obs.values())[0]
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        value_and_logits = self.fc3(h2)

        return value_and_logits[..., 0], value_and_logits[..., 1:]


if __name__ == "__main__":
    set_process_logger(file_path="./log/server.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelNet(8, 4).to(device)
    mr = MultiProcessBatcher(maxlen=3000, device=device, batch_size=64, forward_steps=64, num_batch_maker=1)
    learner = IMPALA(model, mr, lr=1e-3, ef=3e-5, vf=0.5)
    learner_server = ActorServer(learner, port=8010)
    learner_server.run()

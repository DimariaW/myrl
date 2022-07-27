import gfootball.env as gfootball_env
from tests.football.football_env import CHWWrapper
from tests.football.football_model import CNNModel
from myrl.agent import IMPALAAgent

import myrl.core as core
import torch


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self, gather_id: int, actor_id: int):
        env = gfootball_env.create_environment(env_name="11_vs_11_easy_stochastic",
                                               stacked=True,
                                               rewards="scoring,checkpoints",
                                               render=False,
                                               representation="extracted")
        env = CHWWrapper(env)
        device = torch.device("cpu")
        model = CNNModel(env.observation_space.shape, env.action_space.n).to(device)
        agent = IMPALAAgent(model, device)
        return env, agent


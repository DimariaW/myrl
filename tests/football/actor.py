import bz2
import pickle

import gfootball.env as gfootball_env
from tests.football.football_env import CHWWrapper
from tests.football.football_model import CNNModel
from myrl.agent import IMPALAAgent

import myrl.core as core
import torch


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self, gather_id: int = None, actor_id: int = None):
        env = gfootball_env.create_environment(env_name="11_vs_11_hard_stochastic",
                                               stacked=True,
                                               rewards="scoring,checkpoints",
                                               render=False,
                                               representation="extracted")
        env = CHWWrapper(env)
        device = torch.device("cpu")
        model = CNNModel(env.observation_space.shape, env.action_space.n).to(device)
        agent = IMPALAAgent(model, device)
        return env, agent


"""
if __name__ == "__main__":
    from myrl.actor import Actor
    env, agent = ActorCreate().create_env_and_agent()
    actor = Actor(env, agent, 32, get_full_episode=False)
    episodes = []
    episodes_compressed = []
    for _ in range(4):
        episode = actor.sample()
        episodes.append(episode)
        episodes_compressed.append(bz2.compress(pickle.dumps(episode)))
    assert(True)
"""

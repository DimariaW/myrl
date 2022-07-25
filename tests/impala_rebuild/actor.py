import gym
import torch

import envs.env_wrapper as env_wrapper
import myrl.utils as utils
import myrl.train as core
from tests.a2c_rebuild.model import Model
from myrl.agent import IMPALAAgent


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self):
        env = gym.make("LunarLander-v2")
        env = env_wrapper.ScaleReward(env, 1 / 200)
        env = env_wrapper.DictObservation(env, "feature")

        device = torch.device("cpu")
        model = Model(8, 4).to(device)
        agent = IMPALAAgent(model, device)
        return env, agent





"""
if __name__ == "__main__":
    utils.set_process_logger()
    env = gym.make("LunarLander-v2")
    env = env_wrapper.ScaleReward(env, 1/200)
    env = env_wrapper.DictObservation(env, "feature")
    device = torch.device("cpu")
    model = Model(8, 4).to(device)
    agent = PGAgent(model, device)
    actor = Actor(env, agent, steps=256, get_full_episode=True)
    episode1 = actor.sample()
    episode2 = actor.sample()
    episode3 = actor.sample()
    actor.predict()
    actor.predict()
    episode4 = actor.sample()
    actor.predict()
    assert(True)
"""



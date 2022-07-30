import gym
import torch
from tests.a2c_rebuild.model import Model
from myrl.agent import ACAgent
import gym
import torch

import envs.env_wrapper as env_wrapper

import myrl.core as core


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self, gather_id: int, actor_id: int):
        env = gym.make("LunarLander-v2")
        env = env_wrapper.ScaleReward(env, 1/200)
        env = env_wrapper.DictReward(env)
        #env = env_wrapper.DictObservation(env, "feature")
        device = torch.device("cpu")
        model = Model(8, 4, use_orthogonal_init=True, use_tanh=False).to(device)
        agent = ACAgent(model, device)
        return env, agent




"""
if __name__ == "__main__":
    utils.set_process_logger()
    env = gym.make("LunarLander-v2")
    env = env_wrapper.ScaleReward(env, 1/200)
    env = env_wrapper.DictObservation(env, "feature")
    device = torch.device("cpu")
    model = Model(8, 4).to(device)
    agent = ACAgent(model, device)
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



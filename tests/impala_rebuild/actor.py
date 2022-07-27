import gym
import torch

import envs.env_wrapper as env_wrapper

import myrl.core as core

from tests.impala_rebuild.model import Model

from myrl.agent import IMPALAAgent


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self, gather_id: int, actor_id: int):
        env = gym.make("LunarLander-v2")
        env = env_wrapper.ScaleReward(env, 1/200)
        #env = env_wrapper.DictObservation(env, "feature")

        device = torch.device("cpu")
        model = Model(8, 4, use_orthogonal_init=True, use_tanh=False).to(device)
        agent = IMPALAAgent(model, device)
        return env, agent



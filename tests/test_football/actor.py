import gfootball.env as gfootball_env
from envs.football.football_env import TamakEriFeverEnv, SimpleEnv
from envs.football.football_model import FootballNet, SimpleModel

from myrl.agent import IMPALAAgent
from myrl.actor import Actor, ActorClient, open_gather
from myrl.utils import set_process_logger

import torch


def create_actor(actor_indexes: tuple, queue_gather2actor, queue_actor2gather):
    actor_index, num_samples, num_evals = actor_indexes

    set_process_logger(file_path=f"./log/11_vs_11_easy_stochastic/actor_{actor_index}.txt")
    env = gfootball_env.create_environment(env_name="academy_empty_goal",
                                           representation="raw",
                                           rewards="scoring")
    env = TamakEriFeverEnv(env)
    #env = SimpleEnv(env)

    device = torch.device("cpu")
    model = FootballNet().to(device)
    #model = SimpleModel(2, 1).to(device)
    agent = IMPALAAgent(model, device)
    if actor_index < num_samples:
        actor = Actor(env, agent, steps=64, get_full_episode=False)
        actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="sampler")
    else:
        actor = Actor(env, agent, steps=64, get_full_episode=True)
        actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="evaluator")
    actor_client.run()


import gfootball.env as gfootball_env
from envs.football.football_env import TamakEriFeverEnv, SimpleEnv
from envs.football.football_model import FootballNet, SimpleModel

from myrl.agent import IMPALAAgent
from myrl.actor import Actor, ActorClient, open_gather
from myrl.utils import set_process_logger

import torch


def create_actor(actor_index: int, queue_gather2actor, queue_actor2gather):
    set_process_logger(file_path=f"./log/empty_goal/actor_{actor_index}.txt")
    env = gfootball_env.create_environment(env_name="academy_empty_goal",
                                           representation="raw",
                                           rewards="scoring,checkpoints")
    env = TamakEriFeverEnv(env)
    #env = SimpleEnv(env)

    device = torch.device("cpu")
    model = FootballNet().to(device)
    #model = SimpleModel(2, 1).to(device)
    agent = IMPALAAgent(model, device)
    if actor_index < 5:
        actor = Actor(env, agent, steps=128, get_full_episodes=False)
        actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="sampler")
    else:
        actor = Actor(env, agent, steps=128, get_full_episodes=True)
        actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="evaluator")
    actor_client.run()


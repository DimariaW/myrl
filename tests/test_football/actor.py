import gfootball.env as gfootball_env
from envs.football.football_env import SimpleEnv
from envs.football.football_model import SimpleModel

from myrl.agent import IMPALAAgent
from myrl.actor import Actor, ActorClient, open_gather
from myrl.utils import set_process_logger

import torch


def create_actor(actor_index: int, queue_gather2actor, queue_actor2gather):
    set_process_logger()
    env = gfootball_env.create_environment(env_name="academy_empty_goal",
                                           representation="raw",
                                           rewards="scoring")
    env = SimpleEnv(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(2, 1).to(device)
    agent = IMPALAAgent(model, device)
    actor = Actor(env, agent, steps=128, get_full_episodes=False)
    actor_client = ActorClient(actor_index, actor, queue_gather2actor, queue_actor2gather, role="sampler")
    actor_client.run()


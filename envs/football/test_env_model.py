"""
from football_env import FootballEnv
from football_model_paddle import FootballNet, to_tensor
import tqdm
import paddle

env = FootballEnv()

model = FootballNet()

model.eval()

obs = env.reset()

t_r = 0
for _ in tqdm.tqdm(range(3001)):

    p, r = model(to_tensor(obs, 0))

    obs, rew, done, info = env.step(paddle.argmax(p).item())

    t_r += rew
    if done:
        print(info)
        print(t_r)
"""
#%%
import logging

import gfootball.env as gfootball_env
from envs.football import TamakEriFeverEnv
from envs.football import FootballNet
from myrl.utils import to_tensor, batchify, set_process_logger
import torch
import os
print(os.getcwd())

set_process_logger(stdout_level=logging.DEBUG)


def load_model(model, model_path):
    loaded_dict_ = torch.load(model_path)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict_.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)
    return model


device = torch.device("cpu")
env = gfootball_env.create_environment(env_name="11_vs_11_kaggle_level1", representation="raw", rewards="scoring,checkpoints")
env = TamakEriFeverEnv(env)

net = FootballNet()
net.load_state_dict(torch.load("./1679.pth"), strict=True)
net.eval()


obs = env.reset()
logging.info("hello")


def infinite():
    while True:
        yield 0
#%%
from tqdm import tqdm

for _ in tqdm(infinite()):
    obs_tensor = to_tensor(batchify([batchify([obs], unsqueeze=0)], unsqueeze=0), unsqueeze=None, device=device)
    _, logit = net(obs_tensor)

    obs, reward, done, info = env.step(torch.argmax(logit).item())
    if done:
        logging.info(info)
        obs = env.reset()

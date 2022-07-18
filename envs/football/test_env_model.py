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
import gfootball.env as gfootball_env
from envs.football import TamakEriFeverEnv
from envs.football import FootballNet
from myrl.utils import to_tensor, batchify
import torch


def load_model(model, model_path):
    loaded_dict_ = torch.load(model_path)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict_.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)
    return model


device = torch.device("cpu")
env = gfootball_env.create_environment(env_name="11_vs_11_kaggle", representation="raw")
env = TamakEriFeverEnv(env)

net = FootballNet()
net = load_model(net, "./1679.pth")
net.eval()


obs = env.reset()

def infinite():
    while True:
        yield 0
#%%
from tqdm import tqdm

for _ in tqdm(infinite()):
    _, logit = net(to_tensor(batchify([obs], unsqueeze=0), unsqueeze=None, device=device))

    obs, reward, done, info = env.step(torch.argmax(logit).item())
    if done:
        print(info)
        obs = env.reset()

import gfootball.env as gfootball_env
from envs.football.football_env import SimpleEnv
from envs.football.football_model import SimpleModel
from myrl.utils import to_tensor, batchify
import torch

env = gfootball_env.create_environment(env_name="academy_empty_goal_close",
                                       representation="raw",
                                       rewards="scoring,checkpoints")
env = SimpleEnv(env)
model = SimpleModel(2, 1)

obs = env.reset()

t = 0
while True:
    v, logit = model(to_tensor(batchify([obs], unsqueeze=0), unsqueeze=None))
    print(v)
    action = torch.argmax(logit).item()
    obs, reward, done, info = env.step(action)
    t += reward
    if done:
        print(t)
        break
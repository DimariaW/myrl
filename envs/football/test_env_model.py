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

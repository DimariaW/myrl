import gfootball.env as football_env
import envs.football.rulebaseA as rulebaseA
import envs.football.rulebaseB as rulebaseB
import envs.football.rulebaseC as rulebaseC

import tqdm

env = football_env.create_environment(
    env_name="11_vs_11_kaggle",
    stacked=False,
    representation="raw",
    #logdir="./videos/",
    #write_goal_dumps=False,
    #write_full_episode_dumps=True,
    #write_video=False,
    render=False,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=1,
    other_config_options={'video_format': 'webm',
                          'action_set': 'v2'}
)

obs = env.reset()

for i in tqdm.tqdm(range(3001)):
    obs, reward, done, info = env.step([rulebaseC.agent(obs[0]), 19])
    if done:
        print(obs[0]["score"])
        print(i)
        break





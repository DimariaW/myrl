import gfootball.env as gfootball_env
import gym
import numpy as np


class CHWWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # HWC
        obs_shape = env.observation_space.shape
        # CHW
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


if __name__ == "__main__":
    env_ = gfootball_env.create_environment(env_name="11_vs_11_easy_stochastic",
                                            stacked=True,
                                            rewards="scoring,checkpoints",
                                            render=False,
                                            representation="extracted")
    env_ = CHWWrapper(env_)
    obs_ = env_.reset()
    obs, reward, done, info = env_.step(0)
    assert(True)


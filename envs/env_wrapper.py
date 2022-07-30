import gym
import numpy as np


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, scale_factor: float = 1):
        super().__init__(env)
        self.scale_factor = scale_factor

    def reward(self, reward):
        return reward * self.scale_factor


class DictObservation(gym.ObservationWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self.key = key

    def observation(self, observation: np.ndarray):
        return {self.key: observation}


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class RunningMeanStdV2:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.square_mean = np.zeros(shape)

    def update(self, x: np.ndarray):
        self.n += 1
        self.mean = self.mean + (x - self.mean) / self.n
        self.square_mean = self.square_mean + (np.square(x) - self.square_mean) / self.n

    @property
    def std(self):
        return np.sqrt(self.square_mean - np.square(self.mean)) if self.n > 1 else np.ones_like(self.mean)


class NormObservation(gym.ObservationWrapper):
    def __init__(self, env, obs_shape):
        super().__init__(env)
        self.normalizer = RunningMeanStdV2(obs_shape)

    def observation(self, observation):
        self.normalizer.update(observation)
        return (observation - self.normalizer.mean) / (self.normalizer.std + 1e-8)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStdV2(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class ScaleRunningReward(gym.Wrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
        self.reward_scaler = RewardScaling(1, self.gamma)

    def reset(self):
        obs = self.env.reset()
        self.reward_scaler.reset()

    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        return obs, self.reward_scaler(reward), done, info


class DictReward(gym.RewardWrapper):
    def reward(self, reward):
        return {"reward": reward}


if __name__ == "__main__":
    x = np.random.randn(100, 3)
    m1 = RunningMeanStd(3)
    m2 = RunningMeanStdV2(3)
    for i in range(100):
        m1.update(x[i])
        m2.update(x[i])
        mean = x[:i+1].mean(axis=0)
        assert(np.sum(m1.mean - m2.mean) <= 1e-5)
        assert(np.sum(m2.mean - mean) <= 1e-5)
        std = x[:i + 1].std(axis=0)
        if i != 0:
            assert (np.sum(m1.std - m2.std) <= 1e-5)
            assert (np.sum(m2.std - std) <= 1e-5)



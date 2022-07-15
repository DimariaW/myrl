import gym


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

    def observation(self, observation):
        return {self.key: observation}
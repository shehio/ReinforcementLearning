import numpy as np


class Action:

    def __init__(self, reward, to, probability=1): # Todo: Validate state here too.
        if not isinstance(reward, (int, float, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)):
            raise TypeError("Reward has to be a number.")
        self.reward = reward
        self.to = to
        self.probability = probability

    def get_value(self, discount_factor: float):
        return self.reward + discount_factor * self.to.update_value

    def __repr__(self):
        return f'Reward: {self.reward} to state: {self.to.name} with probability: {self.probability}.'

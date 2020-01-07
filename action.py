import numpy as np


class Action:

    def __init__(self, reward, to):
        assert to is not None  # TODO: Throw a more specific error
        self.reward = reward
        self.to = to

    def __repr__(self):
        return f'Reward: {self.reward} to state: {self.to.name}.'

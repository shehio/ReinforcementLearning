import numpy as np


class Action:

    def __init__(self, reward, to):
        self.reward = reward
        self.to = to

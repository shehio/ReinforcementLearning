import numpy as np


class MarkovDecisionProcess:

    def __init__(self, state_count):
        self.state_count = state_count
        self.states = np.array(self.state_count)

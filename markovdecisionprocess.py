import numpy as np


class MarkovDecisionProcess:

    def __init__(self, states_count):
        self.states_count = states_count
        self.states = np.ndarray(self.states_count)

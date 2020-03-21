import numpy as np


class MarkovDecisionProcess:

    def __init__(self, states):
        self.states = states
        self.states_count = self.states.size

    # This might not be very efficient.
    def add_state(self, state):
        # verify the uniquness of state and state name.
        self.states = np.append(self.states, state)
        self.states_count = self.states.size

    def __repr__(self):
        returned_string = ''

        for state in self.states:
            returned_string += state.__repr__()

        return returned_string

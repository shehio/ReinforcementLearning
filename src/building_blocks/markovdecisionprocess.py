import numpy as np
from .state import State


class MarkovDecisionProcess:

    def __init__(self, states):
        # @Todo: Verify the type of states.
        self.states = states

    # This might not be very efficient.
    def add_state(self, state):
        if not isinstance(state, State):
            raise TypeError("state has to be of type State.")
        self.states = np.append(self.states, state)

    def contains_state(self, state):
        if not isinstance(state, State):
            raise TypeError("state has to be of type State.")
        return state in self.states

    def __repr__(self):
        returned_string = ''

        for state in self.states:
            returned_string += state.__repr__()

        return returned_string

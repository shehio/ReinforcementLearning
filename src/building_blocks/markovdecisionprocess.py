from .state import State
from .valuefunction import ValueFunction

import numpy as np


class MarkovDecisionProcess:

    def __init__(self, states):  # @Todo: Verify the type of states.
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

    @staticmethod  # This is not right. We shouldn't have access to states here.
    def update_values(value_function: ValueFunction):  # Do we still need the validations with the type hint here?
        any(map(lambda state: state.update_value(value_function.value_dict[state]), value_function.value_dict.keys()))

    def terminal_states(self):
        return sum(state.actions.shape == (0,) for state in self.states)

    def __repr__(self):
        returned_string = ''

        for state in self.states:
            returned_string += state.__repr__()

        return returned_string

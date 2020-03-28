import numpy as np


class State:

    def __init__(self, name, initial_value=0, actions=np.ndarray((0, ))):
        if not isinstance(name, str):
            raise TypeError("Name has to be a string.")
        if not isinstance(initial_value, (int, float)):
            raise TypeError("Value has to be a number.")
        self.name = name
        self.initial_value = initial_value
        self.updated_value = initial_value
        self.actions = actions

    def add_action(self, action):  # Note that it could be a probabilistic action too.
        self.actions = np.append(self.actions, action)

    def update_value(self, value: float):
        self.updated_value = value

    def __repr__(self):
        returned_string = ''
        if self.actions.size == 0:
            returned_string += f'State {self.name} has an initial value of {self.initial_value} and no actions.\n\n'
        else:
            returned_string += f'State {self.name} has an initial value of {self.initial_value}, ' \
                               f'an updated value of {self.updated_value}, and actions:\n'
            for counter, action in enumerate(self.actions):
                returned_string += f'{str(counter + 1)}. {repr(action)}'
            returned_string += '\n\n'

        return returned_string

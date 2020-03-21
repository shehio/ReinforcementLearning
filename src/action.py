from .state import State

class Action:

    def __init__(self, reward, to):
        if not isinstance(to, State):
            raise TypeError("'to' has to be a state.")
        assert to is not None  # TODO: Throw a more specific error
        self.reward = reward
        self.to = to

    def __repr__(self):
        return f'Reward: {self.reward} to state: {self.to.name}.'

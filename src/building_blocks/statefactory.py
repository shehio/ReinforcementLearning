from .action import Action
from .probabilisticaction import ProbabilisticAction
from .state import State


class StateFactory:

    @staticmethod
    def create_state(name, value=0):
        return State(name, value)

    @staticmethod
    def add_action(state, action):  # or probabilisticaction
        if not isinstance(state, State):
            raise TypeError("State to be an instance of State.")
        if not isinstance(action, (Action, ProbabilisticAction)):
            raise TypeError("Action to be an instance of Action or ProbabilisticAction.")
        if not isinstance(action.to, State):
            raise TypeError("'To' in the action has to be an instance of State.")
        state.add_action(action)

    @staticmethod
    def add_actions(state, actions):
        raise NotImplementedError

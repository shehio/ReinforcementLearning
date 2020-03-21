from .action import Action
from .state import State


class StateFactory:

    @staticmethod
    def create_state(name, value):
        return State(name, value)

    @staticmethod
    def add_action(state, action):
        if not isinstance(state, State):
            raise TypeError("State to be an instance of State.")
        if not isinstance(action, Action):
            raise TypeError("Action to be an instance of Action.")
        if not isinstance(action.to, State):
            raise TypeError("'To' in the action has to be an instance of State.")
        state.add_action(action)

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
        if not isinstance(action, ProbabilisticAction):
            raise TypeError("Action to be an instance of Action or ProbabilisticAction.")
        # if isinstance(action, Action) and isinstance(action.to, State):  # This should be part of Action
        #     raise TypeError("To in the action has to be an instance of State.")
        # if isinstance(action, ProbabilisticAction) and isinstance(action.states.dtype, State):  # @Todo: Move to ProbabilisticAction
        #     raise TypeError("The dtype of states in probabilistic_action has to be of type State.")
        state.add_action(action)

    @staticmethod
    def add_actions(state, actions):
        raise NotImplementedError

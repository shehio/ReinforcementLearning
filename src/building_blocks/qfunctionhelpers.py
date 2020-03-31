from .markovdecisionprocess import MarkovDecisionProcess
from .qfunction import QFunction
from .policy import Policy
from .valuefunction import ValueFunction

import functools
import numpy as np


class QFunctionHelpers:

    @staticmethod
    def get_qfunction(mdp: MarkovDecisionProcess, discount_factor: float):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        QFunctionHelpers.__validate_discount_factor(discount_factor)
        state_actions = {}

        # Only get the states that have actions. If not, they should neither be added to QFunction or the value function
        for state in mdp.states:
            if state.actions.shape > (0,):
                state_actions[state] = state.actions

        entry_list = list(
            map(functools.partial(QFunctionHelpers.__get_qvalues, discount_factor), state_actions.items()))
        qdict = {}
        for entry in entry_list:
            qdict.update(entry)

        return QFunction(qdict)

    @staticmethod
    def get_policy_from_max_qfunction(mdp: MarkovDecisionProcess, q_function: QFunction = None):
        if q_function is None:
            q_function = QFunctionHelpers.get_qfunction(mdp)
        dict = dict(map(functools.partial(QFunctionHelpers.__get_max_value_action_from_state, q_function.dict),
                     q_function.dict.keys()))
        return Policy(mdp, dict)

    @staticmethod
    def get_value_function_from_max_qvalue(q_function: QFunction):
        return ValueFunction(
            dict(map(functools.partial(QFunctionHelpers.__get_max_value_from_state, q_function.dict),
                     q_function.dict.keys())))

    @staticmethod
    def __validate_discount_factor(discount_factor):
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor has to be of type float.")
        if discount_factor > 1:
            raise ValueError("discount_factor has to be less or equal one.")

    @staticmethod
    def __get_max_value_action_from_state(qdict, state):
        values = qdict[state].values()
        amax = np.argmax(qdict[state].values())
        return state, qdict[state].keys()[amax]

    @staticmethod
    def __get_max_value_from_state(qdict, state):
        values = qdict[state].values()
        return state, max(qdict[state].values())

    @staticmethod
    def __get_qvalues(discount_factor: float, state_actions_tuple):
        state = state_actions_tuple[0]
        actions = state_actions_tuple[1]
        state_action_values = {}
        state_action_values[state] = {}
        for action in actions:
            state_action_values[state][action] = action.get_value(discount_factor)
        # print(state_action_values)
        return state_action_values
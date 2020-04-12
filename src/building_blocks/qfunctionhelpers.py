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

        # Only get the states that have actions. If not, they should not be added to QFunction.
        for state in mdp.states:
            if not state.is_terminal():
                state_actions[state] = state.actions

        get_qvalues = functools.partial(QFunctionHelpers.__get_qvalues, discount_factor)
        entry_list = list(map(get_qvalues, state_actions.items()))

        qdict = {}
        for entry in entry_list:
            qdict.update(entry)
        return QFunction(qdict)

    @staticmethod  # Greedy approach.
    def get_policy_using_max_qfunction_from_mdp(
            mdp: MarkovDecisionProcess,
            discount_factor,
            q_function: QFunction = None):
        QFunctionHelpers.__validate_discount_factor(discount_factor)
        if q_function is None:
            q_function = QFunctionHelpers.get_qfunction(mdp, discount_factor)
        get_max_value_action_from_state = functools.partial(
            QFunctionHelpers.__get_max_value_action_from_state,
            q_function.qdict)
        policy_dict = dict(map(get_max_value_action_from_state, q_function.qdict.keys()))
        return Policy(mdp, policy_dict)

    @staticmethod
    def get_value_function_from_max_qvalue(q_function: QFunction):
        get_max_value_from_state = functools.partial(QFunctionHelpers.__get_max_value_from_state, q_function.qdict)
        return ValueFunction(dict(map(get_max_value_from_state, q_function.qdict.keys())))

    @staticmethod
    def __validate_discount_factor(discount_factor):
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor has to be of type float.")
        if discount_factor > 1:
            raise ValueError("discount_factor has to be less or equal one.")

    @staticmethod
    def __get_max_value_action_from_state(qdict, state):
        values = qdict[state].values()
        amax = np.argmax(list(qdict[state].values()))
        return state, list(qdict[state].keys())[amax]

    @staticmethod
    def __get_max_value_from_state(qdict, state):
        values = qdict[state].values()
        return state, max(qdict[state].values())

    @staticmethod
    def __get_qvalues(discount_factor: float, state_actions_tuple):
        state = state_actions_tuple[0]
        actions = state_actions_tuple[1]
        state_action_values: dict = {state: {}}

        for action in actions:
            state_action_values[state][action] = action.get_value(discount_factor)

        return state_action_values

from .markovdecisionprocess import MarkovDecisionProcess
from .qfunction import QFunction
from .valuefunction import ValueFunction

import functools
import numpy as np


class ValueFunctionHelpers:

    @staticmethod
    def create_value_function(mdp: MarkovDecisionProcess):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        return ValueFunction(dict(map(lambda state: (state, state.value), mdp.states)))

    @staticmethod
    def value_iteration(mdp: MarkovDecisionProcess, discount_factor: float, epsilon: float):
        prev_value_function = 0
        value_function_diff = float("inf")

        while value_function_diff > epsilon:
            q_function = ValueFunctionHelpers.__evaluate_qvalues(mdp, discount_factor)
            current_value_function = ValueFunctionHelpers.__get_value_function(q_function)
            value_function_diff = ValueFunctionHelpers.__get_value_function_difference(
                prev_value_function,
                current_value_function)

            mdp.update_values(current_value_function)
            prev_value_function = current_value_function

        return current_value_function

    @staticmethod
    def __validate_discount_factor(discount_factor):
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor has to be of type float.")
        if discount_factor > 1:
            raise ValueError("discount_factor has to be less or equal one.")

    @staticmethod
    def __get_value_function(q_function: QFunction):
        return ValueFunction(
            dict(map(lambda state: (state, max(q_function.dict[state].values)), q_function.dict.keys())))

    @staticmethod
    def __evaluate_qvalues(mdp: MarkovDecisionProcess, discount_factor):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        ValueFunctionHelpers.__validate_discount_factor(discount_factor)
        state_actions = dict(map(lambda state: (state, state.actions), mdp.states))
        return QFunction(
            dict(map(functools.partial(ValueFunctionHelpers.__get_qvalues, discount_factor), state_actions.items())))

    @staticmethod
    def __get_qvalues(state_actions_tuple, discount_factor):
        state = state_actions_tuple[0]
        actions = state_actions_tuple[1]
        return dict(map(lambda action: ((state, (action, action.get_value(discount_factor))), actions)))

    @staticmethod
    def __get_value_function_difference(first_value_function: ValueFunction, second_value_function: ValueFunction):
        first_vf_keys = first_value_function.dict.keys()
        second_vf_keys = second_value_function.dict.keys()

        if len(first_vf_keys) != len(second_vf_keys):
            raise ValueError("The numbers of states of the first and the second value functions are different.")
        if len(np.intersect1d(first_vf_keys, second_vf_keys)) != len(first_vf_keys):
            raise ValueError("There seems to be different states between the value functions.")

        state_differences = list(
            map(lambda state: abs(first_value_function.dict[state] - second_value_function.dict[state]), first_vf_keys))
        return max(state_differences)

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
        return ValueFunction(dict(map(lambda state: (state, state.updated_value), mdp.states)))

    # Fix this API
    @staticmethod  # How to make iterations an int while passing a default value?
    def value_iteration(
            mdp: MarkovDecisionProcess,
            discount_factor: float,
            epsilon: float,
            iterations: 100,
            terminal_states: 0):
        prev_value_function = ValueFunctionHelpers.create_value_function(mdp)
        value_function_diff = float("inf")

        i = 0
        while i <  2: ##iterations or value_function_diff > epsilon:
            q_function = ValueFunctionHelpers.__evaluate_qvalues(mdp, discount_factor)
            # print(q_function)
            current_value_function = ValueFunctionHelpers.__get_value_function(q_function)
            print('================= CURRENT VALUE FUNCTION =================')
            print(current_value_function)
            value_function_diff = ValueFunctionHelpers.__get_value_function_difference(
                prev_value_function,
                current_value_function,
                terminal_states)  # The first iteration has an extra state. All other states are normalized.

            mdp.update_values(current_value_function)
            print('================= MDP =================')
            print(mdp)
            prev_value_function = current_value_function
            i = i + 1
            terminal_states = 0

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
            dict(map(functools.partial(ValueFunctionHelpers.get_value_from_state, q_function.dict),
                     q_function.dict.keys())))

    @staticmethod
    def get_value_from_state(qdict, state):
        values = qdict[state].values()
        return state, max(qdict[state].values())

    @staticmethod
    def __evaluate_qvalues(mdp: MarkovDecisionProcess, discount_factor: float):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        ValueFunctionHelpers.__validate_discount_factor(discount_factor)
        state_actions = {}

        # Only get the states that have actions. If not, they should neither be added to QFunction or the value function
        for state in mdp.states:
            if state.actions.shape > (0,):
                state_actions[state] = state.actions

        entry_list = list(
            map(functools.partial(ValueFunctionHelpers.__get_qvalues, discount_factor), state_actions.items()))
        qdict = {}
        for entry in entry_list:
            qdict.update(entry)

        return QFunction(qdict)

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

    @staticmethod
    def __get_value_function_difference(
            first_value_function: ValueFunction,
            second_value_function: ValueFunction,
            states_without_actions_count: int):

        first_vf_keys = first_value_function.dict.keys()
        second_vf_keys = second_value_function.dict.keys()

        if abs(len(first_vf_keys) - len(second_vf_keys)) != states_without_actions_count:
            raise ValueError("The numbers of states of the first and the second value functions are different.")
        if len(np.intersect1d(first_vf_keys, second_vf_keys)) == len(first_vf_keys):
            raise ValueError("There seems to be different states between the value functions.")

        if len(first_vf_keys) >= len(second_vf_keys):
            minimum_states_vf_keys = second_vf_keys
        else:
            minimum_states_vf_keys = first_vf_keys

        state_differences = list(map(
            lambda state: abs(first_value_function.dict[state] - second_value_function.dict[state]),
            minimum_states_vf_keys))
        return max(state_differences)

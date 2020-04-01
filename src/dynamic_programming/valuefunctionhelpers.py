from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.building_blocks.qfunctionhelpers import QFunctionHelpers
from src.building_blocks.valuefunction import ValueFunction

import numpy as np


## @Todo: Unify the level of abstraction in this class and in qfunctionhelpers.
## @Todo: These helpers shouldn't live in the same folder as the basic components: action, state, mdp.
## @Todo: Rename dict everywhere.
## @Todo: Have a common interface between action and probabilistic action.
class ValueFunctionHelpers:

    @staticmethod
    def create_value_function(mdp: MarkovDecisionProcess):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        return ValueFunction(dict(map(lambda state: (state, state.updated_value), mdp.states)))

    @staticmethod  # How to make iterations an int while passing a default value?
    def value_iteration(
            mdp: MarkovDecisionProcess,
            discount_factor: float,
            iterations: 100,
            epsilon: 0.05):
        prev_value_function = ValueFunctionHelpers.create_value_function(mdp)
        value_function_diff = float("inf")
        terminal_states = mdp.terminal_states()
        i = 0

        while i < iterations and value_function_diff > epsilon:
            q_function = QFunctionHelpers.get_qfunction(mdp, discount_factor)
            current_value_function = QFunctionHelpers.get_value_function_from_max_qvalue(q_function)
            value_function_diff = ValueFunctionHelpers.get_value_function_difference(
                prev_value_function,
                current_value_function,
                terminal_states)  # The first iteration has an extra state. All other states are normalized.

            mdp.update_values(current_value_function)
            print(f'Iteration: {i}')
            print('================= Current MDP =================')
            print(mdp)
            prev_value_function = current_value_function
            i = i + 1
            terminal_states = 0

        policy = QFunctionHelpers.get_policy_using_max_qfunction_from_mdp(mdp, discount_factor)
        return policy, current_value_function

    @staticmethod
    def get_value_function_difference(
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

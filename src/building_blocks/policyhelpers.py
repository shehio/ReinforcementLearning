from .markovdecisionprocess import MarkovDecisionProcess
from .qfunctionhelpers import QFunctionHelpers
from .policy import Policy
from .valuefunctionhelpers import ValueFunctionHelpers

import functools


class PolicyHelpers:

    @staticmethod
    def policy_iteration(mdp: MarkovDecisionProcess, discount_factor, epsilon: 0.05):
        old_policy = Policy(mdp)  # Initializes a random policy by default.
        policy_is_stable = False
        while not policy_is_stable:
            mdp, value_function = PolicyHelpers.evaluate_policy(mdp, old_policy, discount_factor, epsilon)
            new_policy = QFunctionHelpers.get_policy_from_max_qfunction(mdp)
            if PolicyHelpers.are_similar(old_policy, new_policy):
                return old_policy
            old_policy = new_policy

    @staticmethod  # @Todo: Refactor this with the similar code in valuefunctionhelpers.
    def evaluate_policy(mdp: MarkovDecisionProcess, policy: Policy, discount_factor, epsilon: 0.05):
        PolicyHelpers.__validate_input(policy, discount_factor)
        prev_value_function = ValueFunctionHelpers.create_value_function(mdp)
        value_function_diff = float("inf")
        terminal_states = mdp.terminal_states()
        i = 0

        while value_function_diff > epsilon:
            current_value_function = PolicyHelpers.__get_value_function_from_policy(mdp, policy)
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

        return mdp, current_value_function  # @Todo: Don't need the value_function here.

    @staticmethod
    def __validate_input(policy, discount_factor):
        if not isinstance(policy, Policy):
            raise TypeError("policy has to be of type Policy.")
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor has to be of type float.")
        if discount_factor > 1:
            raise ValueError("discount_factor has to be less or equal one.")

    @staticmethod
    def __get_value(state_action_tuple, discount_factor):
        state = state_action_tuple[0]
        action = state_action_tuple[1]
        value = action.get_value(discount_factor)
        return state, value

    # @staticmethod
    # def evaluate_qvalues(policy: Policy, discount_factor):
    #     PolicyHelpers.__validate_input(policy, discount_factor)
    #     return dict(map(functools.partial(PolicyHelpers.__get_qvalue, discount_factor)), policy.__dict.items())

    # @staticmethod
    # def __get_qvalue(state_action_tuple, discount_factor):
    #     state = state_action_tuple[0]
    #     action = state_action_tuple[1]
    #     value = PolicyHelpers.__compute_action_value(action, discount_factor)
    #     return state, action, value


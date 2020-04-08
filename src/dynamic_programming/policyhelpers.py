from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.building_blocks.qfunctionhelpers import QFunctionHelpers
from src.building_blocks.policy import Policy
from src.building_blocks.valuefunction import ValueFunction
from src.dynamic_programming.valuefunctionhelpers import ValueFunctionHelpers

import functools


class PolicyHelpers:

    @staticmethod
    def policy_iteration_sutton(mdp: MarkovDecisionProcess, discount_factor, maximum_iteration=1000, epsilon=0.05):
        new_policy = old_policy = Policy(mdp)  # Initializes a random policy by default.
        policy_is_stable = False
        i = 0
        while not (policy_is_stable or i >= maximum_iteration):
            mdp, value_function = PolicyHelpers.evaluate_policy(mdp, old_policy, discount_factor, epsilon)
            new_policy = QFunctionHelpers.get_policy_using_max_qfunction_from_mdp(mdp, discount_factor)
            if PolicyHelpers.are_similar(old_policy, new_policy):
                policy_is_stable = True
            i = i + 1
            old_policy = new_policy

            PolicyHelpers.__print_current_mdp(i, mdp)

        return new_policy, mdp

    @staticmethod
    def policy_iteration_brunskill(mdp: MarkovDecisionProcess, discount_factor, maximum_iteration=1000, epsilon=0.05):
        policy = Policy(mdp)  # Initializes a random policy by default.
        policy_is_stable = False
        i = 0
        prev_value_function = ValueFunctionHelpers.get_value_function(mdp)

        while not (policy_is_stable or i >= maximum_iteration):
            qfunction = QFunctionHelpers.get_qfunction(mdp, discount_factor)
            current_value_function = QFunctionHelpers.get_value_function_from_max_qvalue(qfunction)
            mdp.update_values(current_value_function)
            policy = QFunctionHelpers.get_policy_using_max_qfunction_from_mdp(mdp, discount_factor, qfunction)

            value_functions_difference = ValueFunctionHelpers.get_value_function_difference(
                prev_value_function,
                current_value_function)
            if value_functions_difference < epsilon:
                policy_is_stable = True

            i = i + 1

            prev_value_function = current_value_function

            PolicyHelpers.__print_current_mdp(i, mdp)

        return policy, mdp

    @staticmethod  # @Todo: Refactor this with the similar code in valuefunctionhelpers.
    def evaluate_policy(mdp: MarkovDecisionProcess, policy: Policy, discount_factor, epsilon: 0.05):
        PolicyHelpers.__validate_input(policy, discount_factor)

        prev_value_function = PolicyHelpers.__get_value_function_from_policy(mdp, policy, discount_factor)
        mdp.update_values(prev_value_function)

        value_function_diff = float("inf")

        while value_function_diff > epsilon: # Should this break anyway after a particular number of iterations?
            current_value_function = PolicyHelpers.__get_value_function_from_policy(mdp, policy, discount_factor)
            value_function_diff = ValueFunctionHelpers.get_value_function_difference(
                prev_value_function,
                current_value_function)

            mdp.update_values(current_value_function)
            prev_value_function = current_value_function

        return mdp, current_value_function

    @staticmethod
    def are_similar(first_policy: Policy, second_policy: Policy):
        if first_policy.mdp != second_policy.mdp:
            return False
        for state in first_policy.policy_dict.keys():
            if PolicyHelpers.__state_is_different_across_policies(
                    state,
                    first_policy.policy_dict,
                    second_policy.policy_dict):
                return False
        return True

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

    @staticmethod
    def __get_value_function_from_policy(mdp: MarkovDecisionProcess, policy: Policy, discount_factor: float):
        return ValueFunction(dict(map(
            functools.partial(PolicyHelpers.__get_state_value, discount_factor, policy),
            mdp.states)))

    @staticmethod
    def __get_state_value(discount_factor, policy, state):
        if state.is_terminal():
            return state, state.updated_value
        val = policy.get_action(state).get_value(discount_factor)
        return state, val

    @staticmethod
    def __print_current_mdp(i, mdp):
        print(f'Iteration: {i}')
        print('================= Current MDP =================')
        print(mdp)
        print()

    @staticmethod
    def __state_is_different_across_policies(state, first_policy_dict, second_policy_dict):
        return state not in second_policy_dict.keys() or first_policy_dict[state] != second_policy_dict[state]

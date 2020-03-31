from .markovdecisionprocess import MarkovDecisionProcess
from .qfunctionhelpers import QFunctionHelpers
from .policy import Policy
from .valuefunction import ValueFunction
from .valuefunctionhelpers import ValueFunctionHelpers

import functools


class PolicyHelpers:

    @staticmethod
    def policy_iteration(mdp: MarkovDecisionProcess, discount_factor, maximum_iteration=1000, epsilon=0.05):
        old_policy = Policy(mdp)  # Initializes a random policy by default.
        policy_is_stable = False
        i = 0
        while not (policy_is_stable or i > maximum_iteration):
            mdp, value_function = PolicyHelpers.evaluate_policy(mdp, old_policy, discount_factor, epsilon)
            new_policy = QFunctionHelpers.get_policy_using_max_qfunction_from_mdp(mdp, discount_factor)
            if PolicyHelpers.are_similar(old_policy, new_policy):
                policy_is_stable = True
            i = i + 1
            old_policy = new_policy

            print(f'Iteration: {i}')
            print('================= Current MDP =================')
            print(mdp)
            print()

        return old_policy

    @staticmethod  # @Todo: Refactor this with the similar code in valuefunctionhelpers.
    def evaluate_policy(mdp: MarkovDecisionProcess, policy: Policy, discount_factor, epsilon: 0.05):
        PolicyHelpers.__validate_input(policy, discount_factor)

        prev_value_function = PolicyHelpers.__get_value_function_from_policy(mdp, policy, discount_factor)
        mdp.update_values(prev_value_function)

        value_function_diff = float("inf")
        terminal_states = 0

        while value_function_diff > epsilon: # Should this break anyway after a particular number of iterations?
            current_value_function = PolicyHelpers.__get_value_function_from_policy(mdp, policy, discount_factor)
            value_function_diff = ValueFunctionHelpers.get_value_function_difference(
                prev_value_function,
                current_value_function,
                terminal_states)  # The first iteration has an extra state. All other states are normalized.

            mdp.update_values(current_value_function)
            prev_value_function = current_value_function

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

    @staticmethod  ## @Todo: bad implementation, defer the fix after creating the unified API between action and paction
    def __get_value_function_from_policy(mdp: MarkovDecisionProcess, policy: Policy, discount_factor: float):
        ## For all the states in the mdp, take the action given by the policy's dict and multiply the discount factor by something.
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
    def are_similar(first_policy: Policy, second_policy: Policy):
        if first_policy.mdp != second_policy.mdp:
            return False
        for state in first_policy.policy_dict.keys():
            if state not in second_policy.policy_dict.keys() or first_policy.policy_dict[state] == second_policy.policy_dict[state]:
                return False
        return True

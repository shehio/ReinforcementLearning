from .action import Action
from .policy import Policy
from .probabilisticAction import ProbabilisticAction

import functools


class PolicyHelpers:

    @staticmethod
    def evaluate_policy(policy: Policy, discount_factor):
        PolicyHelpers.__validate_input(policy, discount_factor)
        return dict(map(functools.partial(PolicyHelpers.__get_value, discount_factor), policy.__dict.items()))

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


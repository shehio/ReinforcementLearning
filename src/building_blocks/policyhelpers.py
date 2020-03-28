from .action import Action
from .policy import Policy
from .probabilisticAction import ProbabilisticAction

import functools


class PolicyHelpers:

    @staticmethod
    def evaluate_policy(policy: Policy, discount_factor):
        if not isinstance(policy, Policy):
            raise TypeError("policy has to be of type Policy.")
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor has to be of type float.")

        return dict(map(functools.partial(PolicyHelpers.__get_value, discount_factor), policy.__dict.items()))

    @staticmethod
    def __get_value(state_action_tuple, discount_factor):
        state = state_action_tuple[0]
        action = state_action_tuple[1]

        if isinstance(action, Action):
            value = action.reward + discount_factor * action.to.value
        elif isinstance(action, ProbabilisticAction):
            value = 0
            for oft_action in action.actions:
                value += oft_action.probability * (oft_action.reward + discount_factor * oft_action.to.value)

        return state, value

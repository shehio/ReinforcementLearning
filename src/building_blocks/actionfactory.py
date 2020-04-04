from .probabilisticaction import ProbabilisticAction
from .state import State

import numpy as np


class ActionFactory:
    @staticmethod
    def create_action(name: str, reward: float, state: State):
        return ProbabilisticAction(name, np.array([reward]), np.array([1.0]), np.array([state]))

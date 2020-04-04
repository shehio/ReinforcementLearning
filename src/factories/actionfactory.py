from src.building_blocks.probabilisticaction import ProbabilisticAction
from src.building_blocks.state import State

import numpy as np


class ActionFactory:
    @staticmethod
    def create_action(name: str, reward: float, state: State):
        return ProbabilisticAction(name, np.array([reward]), np.array([1.0]), np.array([state]))

    @staticmethod
    def create_probabilistic_action(name: str, rewards, probabilities, states):
        return ProbabilisticAction(name, rewards, probabilities, states)

import numpy as np
import random as random
from .action import Action

class ProbabilisticAction:

    def __init__(self, name, rewards, probabilities, states):
        self.__validate_input(name, rewards, probabilities, states)  # Todo: Validate states here too.
        self.name = name
        self.__probabilities = probabilities
        self.cdf = self.__create_cdf(probabilities)
        self.actions =  self.__create_derived_actions(rewards, probabilities, states)

    def get_action(self):
        length = self.__probabilities.shape[0]
        rand = np.random.choice(length, p=self.__probabilities)
        return self.actions[rand]

    def get_value(self, discount_factor: float):
        value = 0
        for oft_action in self.actions:
            value += oft_action.probability * (oft_action.reward + discount_factor * oft_action.to.updated_value)

        return value

    def __validate_input(self, name, rewards, probabilities, states): # Todo: add uniqueness of states test?
        if not isinstance(name, str):
            raise TypeError("Name has to be of type str.")
        self.__validate_rewards(rewards)
        self.__validate_probabilities(probabilities)
        if not rewards.shape == probabilities.shape:
            raise ValueError("Rewards array and probabilities array have different dimensions.")
        if not probabilities.shape == states.shape:
            raise ValueError("Probabilities array and states array have different dimensions.")

    def __validate_rewards(self, rewards):
        if not isinstance(rewards, np.ndarray):
            raise TypeError("Rewards array has to be an ndarray.")
        # https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype
        if not(rewards.dtype in (int, float, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)):
            raise TypeError("The dtype of rewards array has to be an np numerical type.")
        if rewards.size == 0:
            raise ValueError("Rewards array can't be empty.")

    def __validate_probabilities(self, probabilities):
        if not isinstance(probabilities, np.ndarray):
            raise TypeError("Probabilities array has to be an ndarray.")
        if not(probabilities.dtype in (int, float, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)):
            raise TypeError("The dtype of probabilities array has to be an np numerical type.")
        if probabilities.size == 0:
            raise ValueError("Probabilities array can't be empty.")
        if np.sum(probabilities) != 1.0:
            raise ValueError("The sample space probabilities have to sum to 1.0.")

    def __create_derived_actions(self, rewards, probabilities, states):
        return np.asarray(list(map(self.__create_action, rewards, probabilities, states)))

    def __create_action(self, reward, probability, state):
        return Action(reward, state, probability)

    def __create_cdf(self, probabilities):
        return np.cumsum(probabilities)

    def __repr__(self):
        return f'{self.name}'  # 'actions: {self.actions}'

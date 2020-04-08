from src.building_blocks.statediscoveredexception import StateDiscoveredException
from .probabilisticaction import ProbabilisticAction
from .markovdecisionprocess import MarkovDecisionProcess
from .state import State

import random


class Policy:

    def __init__(self, mdp, policy_dict=None):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        self.mdp = mdp

        if policy_dict is None:
            self.policy_dict = self.__create_random_policy()
        else:
            self.policy_dict = policy_dict

    def update_policy(self, state, action):
        self.__validate_state(state)
        if not isinstance(action, ProbabilisticAction):
            raise TypeError("action has to be of type Action.")
        self.policy_dict[state] = action

    def get_action(self, state):
        self.__validate_state(state)
        return self.policy_dict[state]

    def __create_random_policy(self):
        return dict(map(Policy.__get_state_random_action_entry, self.mdp.states))

    def __validate_state(self, state):
        if not isinstance(state, State):
            raise TypeError("state has to be of type State.")
        if not self.mdp.contains_state(state):
            raise StateDiscoveredException()

    @staticmethod
    def __get_state_random_action_entry(state):
        if state.actions.size == 0:
            return state, None
        return state, random.choice(state.actions)

    def __repr__(self):
        returned_string = ''

        for state in self.policy_dict.keys():
            returned_string += f'{state.name} -> {self.policy_dict[state]}\n'

        return returned_string

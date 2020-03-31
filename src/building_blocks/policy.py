from .action import Action
from .markovdecisionprocess import MarkovDecisionProcess
from .state import State

import random


class Policy:

    def __init__(self, mdp):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        self.__mdp = mdp  # @Todo: Do we need a reference of the mdp in here?
        self.__dict = self.__create_random_policy()

    # def __init__(self, mdp, policy_dict):
    #     if not isinstance(mdp, MarkovDecisionProcess):
    #         raise TypeError("mdp has to be of type MarkovDecisionProcess.")
    #     self.__mdp = mdp
    #     self.__dict = policy_dict

    def update_policy(self, state, action):
        self.__validate_state(state)
        if not isinstance(action, Action):
            raise TypeError("action has to be of type Action.")
        self.__dict[state] = action

    def get_action(self, state):
        self.__validate_state(state)
        return self.__dict[state]

    def __create_random_policy(self):
        return dict(map(Policy.__get_state_random_action_entry, self.__mdp.states))

    def __validate_state(self, state):
        if not isinstance(state, State):
            raise TypeError("state has to be of type State.")
        if not self.__mdp.contains_state(state):
            raise ValueError("The mdp for the policy doesn't contain this state.")

    @staticmethod
    def __get_state_random_action_entry(state):
        if state.actions.size == 0:
            return state, None
        return state, random.choice(state.actions)

    def __repr__(self):
        returned_string = ''

        for state in self.__dict.keys():
            returned_string += f'{state.name} -> {self.__dict[state]}\n'

        return returned_string

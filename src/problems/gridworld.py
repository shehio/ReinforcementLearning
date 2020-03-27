import sys
sys.path.append("/Users/shehio/Downloads/workspace/RL")


from src.building_blocks.state import State
from src.building_blocks.statefactory import StateFactory
from src.building_blocks.probabilisticaction import ProbabilisticAction
from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
import numpy as np


# ---------------------------------------
# | state1 | state2 |  state3 | state4  |
# | state5 | ------ |  state6 | state7  |
# | state8 | state9 | state10 | state11 |
# ---------------------------------------
class GridWorld:

    def __init__(self):
        (state1, state2, state3) = GridWorld.create_default_states(('(0, 0)', '(0, 1)',  '(0, 2)'))
        state4 = StateFactory.create_state('(0, 3)', 100)
        (state5, state6) = GridWorld.create_default_states(('(1, 0)', '(1, 2)'))
        state7 = StateFactory.create_state('(1, 3)', -100)
        (state8, state9, state10, state11) = GridWorld.create_default_states(('(2, 0)', '(2, 1)', '(2, 2)', '(2, 3)'))

        probabilities = np.array([0.8, 0.1, 0.1])
        rewards = np.array([0, 0, 0])

        GridWorld.__add_state_transitions(
            state1,
            rewards,
            probabilities,
            np.array(
                [np.array([state2, state1, state5]),
                np.array([state1, state1, state5]),
                np.array([state1, state2, state1]),
                np.array([state5, state2, state1])]))

        GridWorld.__add_state_transitions(
            state2,
            rewards,
            probabilities,
            np.array(
                [np.array([state3, state2, state2]),
                np.array([state1, state2, state2]),
                np.array([state2, state3, state1]),
                np.array([state2, state3, state1])]))

        GridWorld.__add_state_transitions(
            state3,
            rewards,
            probabilities,
            np.array(
                [np.array([state4, state3, state6]),
                np.array([state2, state3, state6]),
                np.array([state3, state4, state2]),
                np.array([state6, state4, state2])]))

        GridWorld.__add_state_transitions(
            state5,
            rewards,
            probabilities,
            np.array(
                [np.array([state5, state1, state8]),
                np.array([state5, state1, state8]),
                np.array([state1, state5, state5]),
                np.array([state8, state5, state5])]))

        GridWorld.__add_state_transitions(
            state6,
            rewards,
            probabilities,
            np.array(
                [np.array([state7, state3, state10]),
                 np.array([state6, state3, state10]),
                 np.array([state3, state7, state6]),
                 np.array([state10, state7, state6])]))

        GridWorld.__add_state_transitions(
            state7,
            rewards,
            probabilities,
            np.array(
                [np.array([state7, state4, state11]),
                 np.array([state6, state4, state11]),
                 np.array([state4, state7, state6]),
                 np.array([state11, state7, state6])]))

        GridWorld.__add_state_transitions(
            state8,
            rewards,
            probabilities,
            np.array(
                [np.array([state9, state5, state8]),
                 np.array([state8, state5, state8]),
                 np.array([state5, state9, state5]),
                 np.array([state8, state9, state8])]))

        GridWorld.__add_state_transitions(
            state9,
            rewards,
            probabilities,
            np.array(
                [np.array([state10, state9, state9]),
                 np.array([state8, state9, state9]),
                 np.array([state9, state10, state8]),
                 np.array([state9, state10, state8])]))

        GridWorld.__add_state_transitions(
            state10,
            rewards,
            probabilities,
            np.array(
                [np.array([state11, state6, state10]),
                 np.array([state9, state6, state10]),
                 np.array([state6, state11, state9]),
                 np.array([state10, state11, state9])]))

        GridWorld.__add_state_transitions(
            state11,
            rewards,
            probabilities,
            np.array(
                [np.array([state11, state7, state11]),
                 np.array([state10, state7, state11]),
                 np.array([state7, state11, state10]),
                 np.array([state11, state11, state10])]))

        states = np.array([state1, state2, state3, state4, state5, state6, state7, state8, state9, state10, state11])
        self.mdp = MarkovDecisionProcess(states)

        print(self.mdp)

    @staticmethod
    def create_default_states(names_list):
        return list(map(StateFactory.create_state, names_list))

    @staticmethod
    def __add_state_transitions(state, rewards, probabilities, transitions):  # right, left, up, down
        action_names = ['right', 'left', 'up', 'down']
        for (action_name, transition) in zip(action_names, transitions):
            probabilistic_action = ProbabilisticAction(action_name, rewards, probabilities, transition)
            StateFactory.add_action(state, probabilistic_action)


GridWorld()
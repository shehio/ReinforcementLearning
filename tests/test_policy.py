from src.building_blocks.action import Action
from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.building_blocks.policy import Policy
from src.building_blocks.state import State

import numpy as np


state_count = 0
states = np.ndarray(state_count, dtype=State)


def test_print():
    mdp = MarkovDecisionProcess(states)
    state0 = State('state 0', 0)
    state1 = State('state 1', 0)
    state2 = State('state 2', 0)

    action1 = Action(5, state1)

    state0.add_action(action1)
    state2.add_action(action1)

    mdp.add_state(state0)
    mdp.add_state(state1)
    mdp.add_state(state2)

    policy = Policy(mdp)

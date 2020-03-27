from src.building_blocks import action as action_module, state as state_module, markovdecisionprocess as mdp_module
from src.building_blocks import policy as policy_module
import numpy as np


state_count = 0
states = np.ndarray(state_count, dtype=state_module.State)


def test_print():
    mdp = mdp_module.MarkovDecisionProcess(states)
    state0 = state_module.State('state 0', 0)
    state1 = state_module.State('state 1', 0)
    state2 = state_module.State('state 2', 0)

    action1 = action_module.Action(5, state1)

    state0.add_action(action1)
    state2.add_action(action1)

    mdp.add_state(state0)
    mdp.add_state(state1)
    mdp.add_state(state2)

    policy = policy_module.Policy(mdp)

import markovdecisionprocess as mdp_module
import state as state_module
import numpy as np


state_count = 5
states = np.ndarray(state_count)


def test_init():
    mdp = mdp_module.MarkovDecisionProcess(states)
    assert mdp is not None
    assert state_count == mdp.states_count
    assert (state_count, ) == mdp.states.shape
    assert states.all() == mdp.states.all()


def test_add_state():
    mdp = mdp_module.MarkovDecisionProcess(states)
    state = state_module.State('state', 0)
    mdp.add_state(state)
    assert (state_count + 1, ) == mdp.states.shape
    assert state == mdp.states[state_count]

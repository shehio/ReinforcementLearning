from src.building_blocks import action as action_module, state as state_module, markovdecisionprocess as mdp_module
import numpy as np


state_count = 0
states = np.ndarray(state_count, dtype=state_module.State)


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


def test_print():  # TODO: Test other combinations
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

    assert mdp.__repr__() == 'Actions from state state 0:\n' + \
        'Reward: 5 to state: state 1.\n' + \
        'There are no actions from state state 1.\n' + \
        'Actions from state state 2:\n' + \
        'Reward: 5 to state: state 1.\n'


test_print()
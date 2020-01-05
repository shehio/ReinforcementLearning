import markovdecisionprocess as mdp_module


def test_init():
    state_count = 5
    mdp = mdp_module.MarkovDecisionProcess(state_count)
    assert mdp is not None
    assert state_count == mdp.states_count
    assert (state_count, ) == mdp.states.shape

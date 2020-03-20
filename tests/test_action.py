import pytest
from src import action as action_module, state as state_module

reward = 5
state = state_module.State('state', 0)
action = action_module.Action(reward, state)


def test_init_with_null():
    with pytest.raises(AssertionError):
        action_module.Action(reward, None)


def test_init():
    assert reward == action.reward
    assert state is action.to


def test_print():
    assert reward == action.reward
    assert 'Reward: 5 to state: state.' == action.__repr__()

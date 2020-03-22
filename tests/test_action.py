import pytest
from src.building_blocks import action as action_module, state as state_module

reward = 5
state = state_module.State('state', 0)
action = action_module.Action(reward, state)


def test_init_with_null_reward():
    with pytest.raises(TypeError) as exception:
        action_module.Action(None, None)
    assert "Reward has to be a number." == str(exception.value)


def test_init_with_non_number_reward():
    with pytest.raises(TypeError) as exception:
        action_module.Action("invalid", None)
    assert "Reward has to be a number." == str(exception.value)


def test_init():
    assert reward == action.reward
    assert state is action.to


def test_print():
    assert reward == action.reward
    assert 'Reward: 5 to state: state with probability: 1.' == action.__repr__()

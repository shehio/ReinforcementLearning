import pytest
import numpy as np
from src.building_blocks import probabilisticaction as action_module, state as state_module

reward = 5
state = state_module.State('state', 0)
action = action_module.Action(reward, state)
probabilities = np.array([1])
rewards = np.array([1])
str_array = np.array(["string"])
empty_int_array = np.array([], dtype=np.int32)
states_array = np.array(state_module.State('state', 0))


def test_init_with_null_rewards():
    with pytest.raises(TypeError) as exception:
        action_module.ProbabilisticAction(None, None, None)
    assert "Rewards array has to be an ndarray." == str(exception.value)


def test_init_with__empty_rewards():
    with pytest.raises(ValueError) as exception:
        action_module.ProbabilisticAction(empty_int_array, None, None)
    assert "Rewards array can't be empty." == str(exception.value)


def test_init_with_non_numerical_rewards():
    with pytest.raises(TypeError) as exception:
        action_module.ProbabilisticAction(str_array, None, None)
    assert "The dtype of rewards array has to be an np numerical type." == str(exception.value)


def test_init_with_null_probabilities():
    with pytest.raises(TypeError) as exception:
        action_module.ProbabilisticAction(rewards, None, None)
    assert "Probabilities array has to be an ndarray." == str(exception.value)


def test_init_with_empty_probabilities():
    with pytest.raises(ValueError) as exception:
        action_module.ProbabilisticAction(rewards, empty_int_array, None)
    assert "Probabilities array can't be empty." == str(exception.value)


def test_init_with_non_numerical_probabilities():
    with pytest.raises(TypeError) as exception:
        action_module.ProbabilisticAction(rewards, str_array, None)
        assert "The dtype of probabilities array has to be an np numerical type." == str(exception.value)


def test_init_with__invalid_probabilities():
    with pytest.raises(ValueError) as exception:
        action_module.ProbabilisticAction(rewards, np.array((0, 0)), None)
    assert "The sample space probabilities have to sum to 1.0." == str(exception.value)


def test_init_with__incompatible_rewards_and_probabilities():
    with pytest.raises(ValueError) as exception:
        action_module.ProbabilisticAction(rewards, np.array((0.5, 0.5)), states_array)
    assert "Rewards array and probabilities array have different dimensions." == str(exception.value)


def test_init_with__incompatible_rewards_and_states():
    with pytest.raises(ValueError) as exception:
        action_module.ProbabilisticAction(np.array((10, 12)), np.array((0.5, 0.5)), states_array)
    assert "Probabilities array and states array have different dimensions." == str(exception.value)

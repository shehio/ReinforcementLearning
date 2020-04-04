from src.building_blocks.probabilisticaction import ProbabilisticAction
from src.building_blocks.state import State

import numpy as np
import pytest

default_name = 'pAction'
default_reward = 5
first_state = State('state', 0)
second_state = State('another_state', 0)
default_probabilities = np.array([1])
rewards = np.array([1])
str_array = np.array(["string"])
empty_int_array = np.array([], dtype=np.int32)
states_array = np.array(State('state', 0))


def test_init_with_null_name():
    with pytest.raises(TypeError) as exception:
        ProbabilisticAction(None, None, None, None)
    assert "Name has to be of type str." == str(exception.value)


def test_init_with_non_string_name():
    with pytest.raises(TypeError) as exception:
        ProbabilisticAction(1, None, None, None)
    assert "Name has to be of type str." == str(exception.value)


def test_init_with_null_rewards():
    with pytest.raises(TypeError) as exception:
        ProbabilisticAction(default_name, None, None, None)
    assert "Rewards array has to be an ndarray." == str(exception.value)


def test_init_with__empty_rewards():
    with pytest.raises(ValueError) as exception:
        ProbabilisticAction(default_name, empty_int_array, None, None)
    assert "Rewards array can't be empty." == str(exception.value)


def test_init_with_non_numerical_rewards():
    with pytest.raises(TypeError) as exception:
        ProbabilisticAction(default_name, str_array, None, None)
    assert "The dtype of rewards array has to be an np numerical type." == str(exception.value)


def test_init_with_null_probabilities():
    with pytest.raises(TypeError) as exception:
        ProbabilisticAction(default_name, rewards, None, None)
    assert "Probabilities array has to be an ndarray." == str(exception.value)


def test_init_with_empty_probabilities():
    with pytest.raises(ValueError) as exception:
        ProbabilisticAction(default_name, rewards, empty_int_array, None)
    assert "Probabilities array can't be empty." == str(exception.value)


def test_init_with_non_numerical_probabilities():
    with pytest.raises(TypeError) as exception:
        ProbabilisticAction(default_name, rewards, str_array, None)
        assert "The dtype of probabilities array has to be an np numerical type." == str(exception.value)


def test_init_with__invalid_probabilities():
    with pytest.raises(ValueError) as exception:
        ProbabilisticAction(default_name, rewards, np.array((0, 0)), None)
    assert "The sample space probabilities have to sum to 1.0." == str(exception.value)


def test_init_with__incompatible_rewards_and_probabilities():
    with pytest.raises(ValueError) as exception:
        ProbabilisticAction(default_name, rewards, np.array((0.5, 0.5)), states_array)
    assert "Rewards array and probabilities array have different dimensions." == str(exception.value)


def test_init_with__incompatible_rewards_and_states():
    with pytest.raises(ValueError) as exception:
        ProbabilisticAction(default_name, np.array((10, 12)), np.array((0.5, 0.5)), states_array)
    assert "Probabilities array and states array have different dimensions." == str(exception.value)


def test_actions_are_computed_correctly():
    probabilistic_action = ProbabilisticAction(
        default_name,
        np.array((10, 12)),
        np.array((0.5, 0.5)),
        np.array((first_state, second_state)))

    __validate_action(probabilistic_action.actions, 0, 10, 0.5, first_state)
    __validate_action(probabilistic_action.actions, 1, 12, 0.5, second_state)


def __validate_action(actions, position, reward, probability, state):
    assert actions[position].reward == reward
    assert actions[position].to == state
    assert actions[position].probability == probability

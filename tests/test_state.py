from src.factories.actionfactory import ActionFactory
from src.building_blocks.probabilisticaction import ProbabilisticAction
from src.building_blocks.state import State

import numpy as np
import pytest

initial_value = 0
state_name = 'state'


def test_init_with_null_name():
    with pytest.raises(TypeError) as exception:
        State(None, initial_value)
    assert "Name has to be a string." == str(exception.value)


def test_init_with_non_string_name():
    with pytest.raises(TypeError) as exception:
        State(23, initial_value)
    assert "Name has to be a string." == str(exception.value)


def test_init_with_null_value():
    with pytest.raises(TypeError) as exception:
        State(state_name, None)
    assert "Value has to be a number." == str(exception.value)


def test_init_with_non_number_value():
    with pytest.raises(TypeError) as exception:
        State(state_name, "invalid")
    assert "Value has to be a number." == str(exception.value)


def test_init():
    state = State(state_name, initial_value)
    assert state is not None
    assert initial_value == state.initial_value
    assert state_name == state.name
    assert (0, ) == state.actions.shape


def test_add_action():
    state = State(state_name, initial_value)
    other_state = State('other_state', initial_value)
    action = ActionFactory.create_action('action', 5, other_state)
    probabilistic_action = ActionFactory.create_probabilistic_action(
        'pAction',
        np.array((10, 12)),
        np.array((0.5, 0.5)),
        np.array((State('third_state', 0), State('fourth_state', 0))))
    state.add_action(action)
    state.add_action(probabilistic_action)

    assert (2, ) == state.actions.shape
    assert action == state.actions[0]
    assert probabilistic_action == state.actions[1]

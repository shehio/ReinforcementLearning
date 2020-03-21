import pytest
from src.building_blocks import action as action_module, state as state_module

initial_value = 0
state_name = 'state'


def test_init_with_null_name():
    with pytest.raises(TypeError) as exception:
        state_module.State(None, initial_value)
    assert "Name has to be a string." == str(exception.value)


def test_init_with_non_string_name():
    with pytest.raises(TypeError) as exception:
        state_module.State(23, initial_value)
    assert "Name has to be a string." == str(exception.value)


def test_init_with_null_value():
    with pytest.raises(TypeError) as exception:
        state_module.State(state_name, None)
    assert "Value has to be a number." == str(exception.value)


def test_init_with_non_number_value():
    with pytest.raises(TypeError) as exception:
        state_module.State(state_name, "invalid")
    assert "Value has to be a number." == str(exception.value)


def test_init():
    state = state_module.State(state_name, initial_value)
    assert state is not None
    assert initial_value == state.value
    assert state_name == state.name
    assert (0, ) == state.actions.shape


def test_add_action():
    state = state_module.State(state_name, initial_value)
    other_state = state_module.State('other_state', initial_value)
    action = action_module.Action(5, other_state)
    state.add_action(action)
    assert (1, ) == state.actions.shape
    assert action == state.actions[0]

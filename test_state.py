import state as state_module
import action as action_module

value = 0
state_name = 'state'


def test_init():
    state = state_module.State(state_name, value)
    assert state is not None
    assert value == state.value
    assert state_name == state.name
    assert (0, ) == state.actions.shape


def test_add_action():
    state = state_module.State(state_name, value)
    action = action_module.Action(5, None)
    state.add_action(action)
    assert (1, ) == state.actions.shape
    assert action == state.actions[0]

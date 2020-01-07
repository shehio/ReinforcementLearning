import state as state_module
import action as action_module

initial_value = 0
state_name = 'state'


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

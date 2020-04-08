from src.factories.actionfactory import ActionFactory
from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.building_blocks.policy import Policy
from src.building_blocks.state import State
from src.building_blocks.statediscoveredexception import StateDiscoveredException

import numpy as np
import pytest


def test_init_with_none_mdp():
    with pytest.raises(TypeError) as exception:
        Policy(None)
    assert "mdp has to be of type MarkovDecisionProcess." == str(exception.value)


def test_init_with_wrong_type_mdp():
    with pytest.raises(TypeError) as exception:
        Policy('invalid')
    assert "mdp has to be of type MarkovDecisionProcess." == str(exception.value)


def test_update_policy_with_none_state():
    mdp = create_toy_mdp()
    policy = Policy(mdp)
    action = ActionFactory.create_action('action', 0, mdp.states[0])
    with pytest.raises(TypeError) as exception:
        policy.update_policy(None, action)
    assert "state has to be of type State." == str(exception.value)


def test_update_policy_with_wrong_type_state():
    mdp = create_toy_mdp()
    policy = Policy(mdp)
    action = ActionFactory.create_action('action', 0, mdp.states[0])
    with pytest.raises(TypeError) as exception:
        policy.update_policy('invalid', action)
    assert "state has to be of type State." == str(exception.value)


def test_update_policy_with_state_not_in_mdp():
    mdp = create_toy_mdp()
    policy = Policy(mdp)
    state = State('invalid', 0)
    action = ActionFactory.create_action('action', 0, mdp.states[0])
    with pytest.raises(StateDiscoveredException) as exception:
        policy.update_policy(state, action)


# def test_update_policy_with_none_action():
#     mdp = create_toy_mdp()
#     policy = Policy(mdp)
#     state = mdp.states[0]
#     with pytest.raises(TypeError) as exception:
#         policy.update_policy(state, None)
#     assert "action has to be of type Action." == str(exception.value)
#
#
# def test_update_policy_with_wrong_type_action():
#     mdp = create_toy_mdp()
#     policy = Policy(mdp)
#     state = mdp.states[0]
#     with pytest.raises(TypeError) as exception:
#         policy.update_policy(state, 'invalid')
#     assert "action has to be of type Action." == str(exception.value)


def test_get_action_with_none_type_state():
    mdp = create_toy_mdp()
    policy = Policy(mdp)
    with pytest.raises(TypeError) as exception:
        policy.get_action(None)
    assert "state has to be of type State." == str(exception.value)


def test_get_action_with_invalid_type_state():
    mdp = create_toy_mdp()
    policy = Policy(mdp)
    state = mdp.states[0]
    with pytest.raises(TypeError) as exception:
        policy.get_action('invalid')
    assert "state has to be of type State." == str(exception.value)


def test_print():
    mdp = create_toy_mdp()

    policy = Policy(mdp)
    policy_string = policy.__repr__()


def create_toy_mdp():
    state0 = State('state 0', 0)
    state1 = State('state 1', 0)
    state2 = State('state 2', 0)

    action0 = ActionFactory.create_action('action0', 7, state0)
    state0.add_action(action0)

    action1 = ActionFactory.create_action('action1', 3, state1)
    state0.add_action(action1)
    state2.add_action(action1)

    action2 = ActionFactory.create_action('action2', 5, state2)
    state0.add_action(action2)

    states = np.array([state0, state1, state2])
    return MarkovDecisionProcess(states)


def validate_toy_mdp(policy_string):
    assert "state 0 ->" in policy_string
    assert "state 1 -> None" in policy_string
    assert "state 2 ->" in policy_string
    assert "Reward: 5" in policy_string or "Reward: 3" in policy_string

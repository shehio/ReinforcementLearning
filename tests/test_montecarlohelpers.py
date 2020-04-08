from src.problems.gridworld import GridWorld
from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.monte_carlo.montecarlohelpers import MonteCarloHelpers
from src.building_blocks.policy import Policy
from src.building_blocks.valuefunction import ValueFunction
from src.dynamic_programming.valuefunctionhelpers import ValueFunctionHelpers

import numpy as np


def test_monte_carlo_prediction_first_visit():
    __test_monte_carlo_prediction(MonteCarloHelpers.monte_carlo_policy_evaluation_first_visit)


def test_monte_carlo_prediction_every_visit():
    __test_monte_carlo_prediction(MonteCarloHelpers.monte_carlo_policy_evaluation_every_visit)


def test_monte_carlo_control_every_visit():
    __test_monte_carlo_control(MonteCarloHelpers.monte_carlo_control_every_visit)


def test_monte_carlo_control_first_visit():
    __test_monte_carlo_control(MonteCarloHelpers.monte_carlo_control_first_visit)


def test_monte_carlo_control_with_discovery_every_visit():
    __test_monte_carlo_control_with_discovery(MonteCarloHelpers.monte_carlo_control_every_visit)


def test_monte_carlo_control_with_discovery_first_visit():
    __test_monte_carlo_control_with_discovery(MonteCarloHelpers.monte_carlo_control_first_visit)


def __test_monte_carlo_prediction(lambda_definition):
    mdp = GridWorld.get_game()
    policy = __create_expected_policy(mdp)
    expected_value_function = __create_expected_value_function(mdp)

    mdp = lambda_definition(
        mdp,
        policy,
        discount_factor=0.9,
        exploration_ratio=0,
        episodes_count=50000)
    actual_value_function = ValueFunctionHelpers.get_value_function(mdp)

    tolerance = 2  # Not based on any statistical analysis
    any(map(
        lambda state: __validate_with_tolerance(
            expected_value_function.value_dict[state],
            actual_value_function.value_dict[state],
            tolerance),
        expected_value_function.value_dict))


def __test_monte_carlo_control(lambda_definition):
    mdp = GridWorld.get_game()
    expected_policy = __create_expected_policy(mdp)
    actual_policy = lambda_definition(
        mdp=mdp,
        discount_factor=0.9,
        stable_count=3,
        exploration_ratio=0.1,
        episodes_count=1000)

    for state in actual_policy.policy_dict.keys():
        assert expected_policy.policy_dict[state] == actual_policy.policy_dict[state]


def __test_monte_carlo_control_with_discovery(lambda_definition):
    mdp = GridWorld.get_game()
    expected_policy = __create_expected_policy(mdp)

    # Remove any states except for the terminal ones.
    mdp.states = np.delete(mdp.states, 4)
    mdp.states = np.delete(mdp.states, 0)
    print(mdp.states)

    actual_policy = lambda_definition(
        mdp=mdp,
        discount_factor=0.9,
        stable_count=3,
        exploration_ratio=0.1,
        episodes_count=5000)

    for state in actual_policy.policy_dict.keys():
        assert expected_policy.policy_dict[state] == actual_policy.policy_dict[state]


def __create_expected_policy(mdp: MarkovDecisionProcess) -> Policy:
    policy_dict = {mdp.states[0]: mdp.states[0].actions[0],
                   mdp.states[1]: mdp.states[1].actions[0],
                   mdp.states[2]: mdp.states[2].actions[0],
                   mdp.states[4]: mdp.states[4].actions[2],
                   mdp.states[5]: mdp.states[5].actions[2],
                   mdp.states[7]: mdp.states[7].actions[2],
                   mdp.states[8]: mdp.states[8].actions[1],
                   mdp.states[9]: mdp.states[9].actions[2],
                   mdp.states[10]: mdp.states[10].actions[1]}
    return Policy(mdp, policy_dict)


def __create_expected_value_function(mdp: MarkovDecisionProcess) -> ValueFunction:
    value_dict = {mdp.states[0]: 64,
                  mdp.states[1]: 74,
                  mdp.states[2]: 85,
                  mdp.states[3]: 100,
                  mdp.states[4]: 57,
                  mdp.states[5]: 57,
                  mdp.states[6]: -100,
                  mdp.states[7]: 49,
                  mdp.states[8]: 43,
                  mdp.states[9]: 48,
                  mdp.states[10]: 28}
    return ValueFunction(value_dict)


def __validate_with_tolerance(expected_value: float, actual_value: float, tolerance: float):
    assert abs(expected_value - actual_value) < tolerance

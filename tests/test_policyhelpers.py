from src.problems.gridworld import GridWorld
from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.building_blocks.policy import Policy
from src.dynamic_programming.policyhelpers import PolicyHelpers


def test_policy_iteration_brunskill():
    __test_policy_iteration_on_grid_world('brunskill')


def test_policy_iteration_sutton():
    __test_policy_iteration_on_grid_world('sutton')
    return


def __test_policy_iteration_on_grid_world(method: str):
    mdp = GridWorld.get_game()
    expected_policy = __create_expected_policy(mdp)

    if method == 'brunskill':
        actual_policy, _ = PolicyHelpers.policy_iteration_brunskill(
            mdp=mdp,
            discount_factor=0.9,
            maximum_iteration=100,
            epsilon=0.1)
    elif method == 'sutton':
        actual_policy, _ = PolicyHelpers.policy_iteration_sutton(
            mdp=mdp,
            discount_factor=0.9,
            maximum_iteration=100,
            epsilon=0.1)

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
from src.problems.gridworld import GridWorld
from src.building_blocks.policyhelpers import Policy
from src.building_blocks.valuefunctionhelpers import ValueFunctionHelpers


def test_value_iteration():
    mdp = GridWorld.get_game()
    policy_dict = {mdp.states[0]: mdp.states[0].actions[0],
                   mdp.states[1]: mdp.states[1].actions[0],
                   mdp.states[2]: mdp.states[2].actions[0],
                   mdp.states[4]: mdp.states[4].actions[2],
                   mdp.states[5]: mdp.states[5].actions[2],
                   mdp.states[7]: mdp.states[7].actions[2],
                   mdp.states[8]: mdp.states[8].actions[1],
                   mdp.states[9]: mdp.states[9].actions[2],
                   mdp.states[10]: mdp.states[10].actions[1]}
    expected_policy = Policy(mdp, policy_dict)
    actual_policy, _ = ValueFunctionHelpers.value_iteration(
        mdp=mdp,
        discount_factor=0.9,
        iterations=100,
        epsilon=0.001)

    for state in actual_policy.policy_dict.keys():
        assert expected_policy.policy_dict[state] == actual_policy.policy_dict[state]

from src.problems.gridworld import GridWorld
from src.building_blocks.policyhelpers import PolicyHelpers


def test_policy_iteration():
    print("starting test")
    mdp = GridWorld.get_game()
    print(mdp)
    PolicyHelpers.policy_iteration(
        mdp=mdp,
        discount_factor=0.9,
        maximum_iteration=1000,
        epsilon=0.1)

test_policy_iteration()
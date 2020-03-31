from src.problems.gridworld import GridWorld
from src.building_blocks.valuefunctionhelpers import ValueFunctionHelpers


def test_value_iteration():
    mdp = GridWorld.get_game()
    ValueFunctionHelpers.value_iteration(
        mdp=mdp,
        discount_factor=0.9,
        iterations=100,
        epsilon=0.001)

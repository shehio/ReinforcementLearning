from src.problems.gridworld import GridWorld
from src.building_blocks.valuefunctionhelpers import ValueFunctionHelpers


def test_value_iteration():
    mdp = GridWorld.get_game()
    ValueFunctionHelpers.value_iteration(mdp, 0.9, 0.001, 2, 2)
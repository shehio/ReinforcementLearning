from src.problems.gridworld import GridWorld
from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.monte_carlo.montecarlohelpers import MonteCarloHelpers
from src.building_blocks.policy import Policy


def test_monte_carlo_first_visit():
    __test_monte_carlo(MonteCarloHelpers.monte_carlo_policy_evaluation_first_visit)


def test_monte_carlo_every_visit():
    __test_monte_carlo(MonteCarloHelpers.monte_carlo_policy_evaluation_every_visit)


def __test_monte_carlo(lambda_definition):
    mdp = GridWorld.get_game()
    policy = __create_expected_policy(mdp)
    discount_factor = 0.9
    simulation_count = 10000

    mdp = lambda_definition(
        mdp,
        policy,
        discount_factor,
        simulation_count)

    print(mdp)


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

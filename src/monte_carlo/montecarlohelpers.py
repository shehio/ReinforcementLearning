from src.building_blocks.markovdecisionprocess import MarkovDecisionProcess
from src.building_blocks.policy import Policy
from src.building_blocks.valuefunction import ValueFunction

import random


class MonteCarloHelpers:

    @staticmethod
    def monte_carlo_policy_evaluation_every_visit(
            mdp: MarkovDecisionProcess,
            policy: Policy,
            discount_factor: float,
            simulation_number: int) -> MarkovDecisionProcess:

        global_counts_dict = {}
        rewards_dict = {}
        i = 0
        while i < simulation_number:
            states_stack, counts_dict, rewards, terminal_state = MonteCarloHelpers.__perform_episode(mdp, policy)
            terminal_reward = terminal_state.updated_value
            MonteCarloHelpers.__absorb_dict(global_counts_dict, counts_dict)

            while len(states_stack) > 0:
                terminal_reward = terminal_reward * discount_factor
                state = states_stack.pop()
                MonteCarloHelpers.__increment_dict(rewards_dict, state, terminal_reward + rewards.pop())

            i = i + 1

        mdp.update_values(ValueFunction(MonteCarloHelpers.__normalize_rewards(rewards_dict, global_counts_dict)))
        return mdp

    @staticmethod
    def monte_carlo_policy_evaluation_first_visit(
            mdp: MarkovDecisionProcess,
            policy: Policy,
            discount_factor: float,
            simulation_number: int) -> MarkovDecisionProcess:

        global_counts_dict = {}
        rewards_dict = {}

        i = 0
        while i < simulation_number:
            states_stack, local_counts_dict, rewards, terminal_state = MonteCarloHelpers.__perform_episode(mdp, policy)
            terminal_reward = terminal_state.updated_value

            while len(states_stack) > 0:
                terminal_reward = terminal_reward * discount_factor
                state = states_stack.pop()
                if local_counts_dict[state] > 1:
                    local_counts_dict[state] = local_counts_dict[state] - 1
                    rewards.pop()
                else:
                    if state in rewards_dict:
                        rewards_dict[state] = rewards_dict[state] + terminal_reward + rewards.pop()
                        global_counts_dict[state] = global_counts_dict[state] + 1
                    else:
                        rewards_dict[state] = terminal_reward + rewards.pop()
                        global_counts_dict[state] = 1

            i = i + 1

        mdp.update_values(ValueFunction(MonteCarloHelpers.__normalize_rewards(rewards_dict, global_counts_dict)))
        return mdp

    @staticmethod
    def __perform_episode(mdp, policy):
        state, states_stack, rewards, counts_dict = MonteCarloHelpers.__initialize_iteration(mdp)
        while not state.is_terminal():
            states_stack.append(state)
            MonteCarloHelpers.__increment_dict(counts_dict, state, 1)
            action_from_policy = policy.get_action(state)
            state, reward = action_from_policy.to()
            rewards.append(reward)
        return states_stack, counts_dict, rewards, state

    @staticmethod
    def __initialize_iteration(mdp):
        state = random.choice(mdp.states)
        states_stack = []
        rewards = []
        counts_dict = {}

        return state, states_stack, rewards, counts_dict

    @staticmethod
    def __increment_dict(dictionary, key, value):
        if key in dictionary:
            dictionary[key] = dictionary[key] + value
        else:
            dictionary[key] = value

    @staticmethod
    def __normalize_rewards(rewards_dict, counts_dict):
        for state in rewards_dict:
            rewards_dict[state] = rewards_dict[state] / counts_dict[state]
        return rewards_dict

    @staticmethod
    def __absorb_dict(global_dict, local_dict):
        for key in local_dict:
            if key not in global_dict:
                global_dict[key] = local_dict[key]
            else:
                global_dict[key] = global_dict[key] + local_dict[key]

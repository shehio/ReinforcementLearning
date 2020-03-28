from .markovdecisionprocess import MarkovDecisionProcess


class ValueFunction:

    def __init__(self, mdp):
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp has to be of type MarkovDecisionProcess.")
        self.dict = self.__create_value_function(mdp)

    def __init__(self, state_values):
        if not isinstance(state_values, dict):
            raise TypeError("state_values has to be of type dict.")
        self.dict = state_values

    @staticmethod
    def __create_value_function(mdp):
        return dict(map(lambda state: (state, state.value), mdp.states))

    def __repr__(self):
        returned_string = ''

        for state in self.__dict.keys():
            returned_string += f'{state.name} -> {self.__dict[state]}\n'

        return returned_string
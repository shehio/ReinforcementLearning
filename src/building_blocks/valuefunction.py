class ValueFunction:

    def __init__(self, state_values):
        if not isinstance(state_values, dict):
            raise TypeError("state_values has to be of type dict.")
        self.dict = state_values

    def __repr__(self):
        returned_string = ''

        for state in self.__dict.keys():
            returned_string += f'{state.name} -> {self.__dict[state]}\n'

        return returned_string
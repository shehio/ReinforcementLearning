class QFunction:

    def __init__(self, state_action_values: dict):
        if not isinstance(state_action_values, dict):
            raise TypeError("state_values has to be of type dict.")
        self.dict = state_action_values

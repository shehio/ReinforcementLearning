class Action:

    def __init__(self, reward, to, probability=1):
        if not isinstance(reward, (int, float)):
            raise TypeError("Reward has to be a number.")
        self.reward = reward
        self.to = to
        self.probability = probability

    def __repr__(self):
        return f'Reward: {self.reward} to state: {self.to.name} with probability: {self.probability}.'

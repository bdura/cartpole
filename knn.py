import numpy as np


class State:
    def __init__(self, observation, action):
        self.observation = observation
        self.action = action
        self.reward = 0

    def still_alive(self):
        self.reward += 1


class States:
    def __init__(self):
        self.elements = []

    def add_state(self, observation, action):
        self.elements += [State(observation, action)]

    def concat(self, states, inplace=False):

        elements = self.elements + states.elements

        if not inplace:
            result = States()
            result.elements = elements
            return result

        self.elements = elements

    def still_alive(self):
        for element in self.elements:
            element.still_alive()

    def lookup(self, observation, k):
        a = np.array([state for state in self.elements if state.action == 0])
        b = np.array([state for state in self.elements if state.action == 1])

        distance_a = np.array([((state.observation - observation)**2).sum() for state in a])
        distance_b = np.array([((state.observation - observation)**2).sum() for state in b])

        indices_a = np.flip(np.argsort(distance_a), 0)[:k]
        indices_b = np.flip(np.argsort(distance_b), 0)[:k]

        return a[indices_a], b[indices_b]


class Agent:
    def __init__(self, k=2, verbose=False):

        self.verbose = verbose
        self.reward = 0

        self.staging = States()
        self.neighborhood = States()

        self.k = k

    def action(self, observation):

        staging = self.staging.lookup(observation, self.k)
        neighborhood = self.neighborhood.lookup(observation, self.k)

        rewards = np.array([
            np.array([state.reward for state in staging[0]] + [state.reward for state in neighborhood[0]] + [1]).mean(),
            np.array([state.reward for state in staging[1]] + [state.reward for state in neighborhood[1]] + [1]).mean()
        ])

        if rewards.min() == rewards.max():
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(rewards)

        self.staging.add_state(observation, action)

        return action

    def add_reward(self, reward):
        self.reward += reward
        if reward == 1:
            self.staging.still_alive()

    def reset(self):
        self.reward = 0
        self.neighborhood.concat(self.staging, inplace=True)
        self.staging = States()

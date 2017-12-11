import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree


class State:
    def __init__(self, observation, previous=None):
        self.observation = observation
        self.reward = 0

        self.previous = previous

    def still_alive(self):
        self.reward += 1


class States:
    def __init__(self):

        self.elements = []
        self.X = pd.DataFrame(columns=['a', 'b', 'c', 'd'])

        self.tree = None

    def compute_tree(self):
        self.tree = KDTree(self.X)

    def add_state(self, observation):
        self.elements += [State(observation)]
        self.X.loc[len(self.X)] = observation

    def concat(self, other, inplace=False):

        elements = self.elements + other.elements
        X = pd.concat([self.X, other.X], ignore_index=True)

        if not inplace:
            result = States()
            result.elements = elements
            result.X = X
            result.compute_tree()
            return result
        else:
            self.elements = elements
            self.X = X
            self.compute_tree()

    def still_alive(self):
        for element in self.elements:
            element.still_alive()

    def lookup(self, observation, k):
        k = min(k, len(self.elements))

        if not self.tree:
            distance = np.array([((state.observation - observation)**2).sum() for state in self.elements])
            indices = np.flip(np.argsort(distance), 0)[:k]
            return distance[indices], [self.elements[index] for index in indices]
        else:
            distance, indices = self.tree.query([observation], k)
            # print(indices)
            return distance, [self.elements[index[0]] for index in indices]


class Path:
    def __init__(self):
        self.states = None


class Agent:
    def __init__(self, k=2, verbose=False):

        self.verbose = verbose
        self.reward = 0

        self.staging = [States(), States()]
        self.neighborhood = [States(), States()]

        self.k = k

    def action(self, observation):

        rewards = []
        distances = []

        for a in [0, 1]:

            distance, states = self.neighborhood[a].lookup(observation, k=self.k)

            if len(distance) == 0:
                rewards.append(1)
                distances.append(1)

            else:
                distance = 1 / distance
                reward = (np.array([state.reward for state in states]) * distance).sum() / distance.sum()

                rewards.append(reward)
                distances.append(distance.mean())

        rewards = np.array(rewards)
        distances = np.array(distances)

        p = (rewards * distances) / ((rewards * distances).sum())

        a = np.random.choice([0, 1], p=p)

        self.staging[a].add_state(observation)

        return a

    def add_reward(self, reward):
        self.reward += reward
        if reward == 1:
            for a in [0, 1]:
                self.staging[a].still_alive()

    def reset(self):
        self.reward = 0
        for a in [0, 1]:
            self.neighborhood[a].concat(self.staging[a], inplace=True)
        self.staging = [States(), States()]

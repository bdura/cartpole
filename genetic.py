import numpy as np


def single_relu(x):

    if x < 0:
        return 0
    else:
        return x


relu = np.vectorize(single_relu)


class Agent:

    def __init__(self, std=0.5, verbose=False):

        self.verbose = verbose
        self.std = std

        self.reward = 0

        self.weights1 = np.random.normal(0, std, [4, 4])
        self.bias1 = .1 * np.ones(4)

        self.weights2 = np.random.normal(0, std, [2, 4])
        self.bias2 = .1 * np.ones(2)

    def print(self, text=''):
        if self.verbose:
            print(text)

    def mutate(self):

        weights1 = self.weights1 + np.random.normal(0, self.std, [4, 4])
        weights2 = self.weights2 + np.random.normal(0, self.std, [2, 4])
        bias1 = self.bias1 + np.random.normal(0, self.std, 4)
        bias2 = self.bias2 + np.random.normal(0, self.std, 2)

        return weights1, bias1, weights2, bias2

    def copy(self):
        weights1, bias1, weights2, bias2 = self.mutate()

        agent = Agent(self.std)

        agent.weights1 = weights1
        agent.weights2 = weights2
        agent.bias1 = bias1
        agent.bias2 = bias2

        return agent

    def action(self, observation):

        result = np.dot(self.weights1, observation)
        self.print(result)

        result += self.bias1
        self.print(result)

        result = relu(result)
        self.print(result)

        result = np.dot(self.weights2, result)
        self.print(result)

        result += self.bias2
        self.print(result)

        return np.argmax(result)

    def full_reward(self, reward):
        self.reward += reward

    def reset(self):
        self.reward = 0


class Generation:

    def __init__(self, n=10, std=.5, verbose=False):

        self.agents = [Agent(std=std, verbose=verbose) for _ in range(n)]

    def reset(self):
        for agent in self.agents:
            agent.reset()

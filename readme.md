# CartPole

A simple project to try out some learning strategies on 
[Open AI Gym](https://gym.openai.com/docs/)'s `CartPole-v0` environment :
* a solution based on genetic algorithm;
* a k nearest-neighbors strategy.


## Genetic Algorithm

The idea is simplistic: make a fixed-sized population of agents compete. 
At every generation, _survivors_ are selected stochastically according to their fitness 
(in this case their overall reward), and mutate into the next generation.

An `Agent` receives information about its environment (the `observation` variable
returned by the `step()` method of the environment), and makes a decision on the 
action to take.

In our case, and `Agent`'s _brain_ consists of a simple two-layer 
neural network, which is fed the observation output by the environment at the end of each step.
The genetic algorithm aims at determining the best parameters for the neural network.


## k-Nearest Neighbors



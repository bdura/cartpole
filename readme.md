# Genetic CartPole

A simple project to try out (asexual) genetic programming on 
[Open AI Gym](https://gym.openai.com/docs/)'s `CartPole
environment.

The idea is simplistic: make a fixed-sized population of agents compete. 
Every _generation_, _survivors_ are selected according to their fitness 
(in this case their overall reward), and mutate into the next generation.

An `Agent` receives information about its environment (the `observation` variable
returned by the `step()` method of the environment), and makes a decision on the 
action to take.

In our case, and `Agent`'s decision process consists of a simple two-layer 
neural network. The genetic algorithm aims at determining the best parameters.
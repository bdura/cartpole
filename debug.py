from genetic import Generation

if __name__ == '__main__':

    gen = Generation(10)

    gen.simulation(10)

    print([agent.reward for agent in gen.agents])

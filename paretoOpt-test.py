import numpy as np

# Mock multifunct for testing (just sum the elements of x)
def multifunct(x, funcao):
    return [sum(x)]  # Simply summing the elements of x

# Define Agente class for the test
class Agente:
    def __init__(self, x=None, funcao=None):
        self.x = x
        self.funcao = funcao
        self.y = multifunct(x, funcao)
        self.rank = -1
    
    def rankme(self, x):
        self.rank = x

    def __str__(self):
        return f"Agente(x={self.x}, y={self.y}, rank={self.rank})"

# Define Populacao class for the test
class Populacao:
    def __init__(self, size, dim, f, crossover_rate, chosen_function):
        self.pop = []
        self.size = size
        self.dim = dim
        self.f = f
        self.cr = crossover_rate
        self.chosen_function = chosen_function
        self.initpop()

    def initpop(self):
        for _ in range(self.size):
            individual = Agente(np.random.rand(self.dim), self.chosen_function)
            self.pop.append(individual)

# Pareto Optimal Function
def paretoOptimal(pop, starting_rank):
    range_n = len(pop.pop[0].y)
    lowest = [float("inf")] * range_n
    pareto = []
    nonpareto = []
    out = True
    for individual in pop.pop:
        if individual.rank >= 0:
            continue
        for i in range(len(individual.y)):
            if individual.y[i] < lowest[i]:
                individual.rankme(starting_rank)
                lowest[i] = individual.y[i]
                out = False
    if out: return
    paretoOptimal(pop, starting_rank+1)

# Test the paretoOptimal function
def test_pareto_optimal():
    size = 10  # Number of individuals
    dim = 3    # Dimensionality of the objective space
    f = 0.5    # A mock mutation factor for testing
    crossover_rate = 0.7
    chosen_function = "sum"  # Mocking the function for the test

    # Create the population
    population = Populacao(size, dim, f, crossover_rate, chosen_function)

    # Test the Pareto Optimal Function
    starting_rank = 0
    population.pop = sorted(population.pop, key=lambda v: tuple(v.y))
    paretoOptimal(population, starting_rank)

    # Print the ranks of the population after Pareto ranking
    print("\nTest Results:")
    for individual in population.pop:
        print(f"Individual: {individual.x} | Rank: {individual.rank} | Objective: {individual.y}")

# Run the test
test_pareto_optimal()

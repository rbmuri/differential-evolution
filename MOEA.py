from random import random

class Individual:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.domination_count = -1
        self.dominated_solutions = []
        
    def dom(self, count):
        self.domination_count = count

    def __lt__(self, other):
        # First compare y in descending order
        if self.y != other.y:
            return self.y > other.y
        # Use x in ascending order as a tiebreaker
        return self.x > other.x
    
    def __repr__(self):
        return f"Individual(x={self.x}, y={self.y})"

def pocalculate(population, dominance):
    """
    This function takes a population of individuals and returns the set of non-dominated individuals.
    """
    poset = []
    population.sort(reverse=True) # sort in relation to y
    bestx = float("inf")
    subpop = []
    for i in population:
        if i.x < bestx:
            bestx = i.x
            i.dom(dominance)
            poset.append(i)
        else:
            subpop.append(i)
    for i in poset:
        print(i)
    return subpop

pop = []     
# for i in range(20):
#     pop.append(Individual(random(), random()))
pop.append(Individual(1, 2))
pop.append(Individual(2, 2))
pop.append(Individual(2, 1))
pop.append(Individual(3, 3))
#pocalculate(pop, 0)
i = 0
while True:
    if not pop: break
    print("SUBSET ", i, "/n")
    pop = pocalculate(pop, i)
    i = i + 1



"""
    pareto_optimal_set = []
    for i in range(len(population)):
        dominated = False
        for j in range(len(population)):
            if i != j and dominates(population[j], population[i]):
                dominated = True
                break
        if not dominated:
            pareto_optimal_set.append(population[i])
    return pareto_optimal_set
"""
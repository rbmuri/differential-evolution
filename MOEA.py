class Individual:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.domination_count = -1
        self.dominated_solutions = []
        
    def dom(self, count):
        self.domination_count += count

def pareto_optimal_set(population):
    """
    This function takes a population of individuals and returns the set of non-dominated individuals.
    """
    pareto_optimal_set = []
    population.sort(reverse=True)
    bestx = float("inf")
    for i in range(len(population)):
        if i.x < bestx:
            bestx = i.x
        while population[i+1].y == population[i].y:
            population[i].dominate(0)

        

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
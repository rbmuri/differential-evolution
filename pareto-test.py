import matplotlib as mpl 
import matplotlib.pyplot as plt

# Assuming paretoOptimal and Agente are already defined and imported
import random

# Generate 50 random (x, y) coordinates
coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)]

# Sort by x, then y
sorted_coords = sorted(coords, key=lambda point: (point[0], point[1]))
print("Sorted Coordinates:")
for point in sorted_coords:
    print(point)

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
                break
            
    if out: return
    paretoOptimal(pop, starting_rank+1)

# Dummy Agente class (if needed standalone)
class Agente:
    def __init__(self, y):
        self.y = y  # Objective function vector
        self.rank = -1
    def rankme(self, r):
        self.rank = r
    def __repr__(self):
        return f"Agente(y={self.y}, rank={self.rank})"

# Dummy Populacao container
class DummyPop:
    def __init__(self, individuals):
        self.pop = individuals

# Create test individuals with 2D objectives
agents = [Agente(list(coord)) for coord in sorted_coords]

# Wrap in DummyPop
test_pop = DummyPop(agents)

# Run the pareto ranking
paretoOptimal(test_pop, 0)

for point in test_pop.pop:
    print(point, " ", point.rank)

ys = []
for i in test_pop.pop:
    if i.rank == 0:
        ys.append(i.y)
fig, ax = plt.subplots()
x_vals, y_vals = zip(*ys)
ax.plot(x_vals, y_vals, color='blue', marker='o')
plt.legend()
plt.show()

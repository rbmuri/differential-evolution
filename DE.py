import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from funct import *

def crossover(CR, n):
    v= np.array()
    for i in range(n):
        if np.random.random() < CR:
            v.append(1)
        else:
            v.append(0)
    return v

class Populacao:
    def __init__(self, f, functiondim, popsize, crossover_rate):
        self.history = []
        self.f = f
        self.dim = functiondim
        self.size = popsize
        self.cr = crossover_rate
        self.y = []
        self.best = float("inf")
    
    def mutate(self, x1, x2, x3, x4):
        res = np.copy(x1)
        for i in range(len(x1)):
            if np.random.random() < self.cr:
                res[i] = x2[i] + self.f * (x3[i] - x4[i])
        return res

    def update_y(self):
        #calculate function results for each
        for i in range(len(self.pop)):
            output = funct(self.pop[i])
            self.y.append(output)
            if output < self.best:
                self.best = output

    def initpop(self):
        self.pop = np.random.random((self.size, self.dim))
        self.update_y()
        self.history.append(self.best)

    def evolve(self, ind):
        #choose four dudes
        chosen = np.random.permutation(self.size)
        #the four dudes are now vectors!
        x1 = self.pop[ind]
        x2 = self.pop[chosen[1]]
        x3 = self.pop[chosen[2]]
        x4 = self.pop[chosen[3]]
        #mutate
        mutation = self.mutate(x1, x2, x3, x4)
        ymutation = funct(mutation)

        if ymutation < self.y[ind]:
            self.pop[ind] = mutation
            self.y[ind] = ymutation
            #update best
            if ymutation < self.best:
                self.best = ymutation

    def run(self, time):
        for t in range(time):
            for i in range(self.size):
                self.evolve(i) 
            self.history.append(self.best)

        
    #prints a line for each individual (with d elements)
    def __str__(self):
        strvalue = ""
        for x in self.pop:
            for y in x:
                strvalue += "(" + str(y) + "), "
            strvalue += "\n"
        return strvalue

pop1 = Populacao(0.8, 3, 50, 0.9)
pop1.initpop()
pop1.run(100)

pop2 = Populacao(0.8, 3, 50, 0.8)
pop2.initpop()
pop2.run(100)

print("\nran! best =", pop1.best, "\n")

fig, ax = plt.subplots()

ax.plot(pop1.history, label="Population 1")
ax.plot(pop2.history, label="Population 2")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Best Value")
plt.title("Differential Evolution")
plt.legend()
plt.show()


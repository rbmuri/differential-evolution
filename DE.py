import numpy as np
import matplotlib as mpl 
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
        self.f = f
        self.functiondim = functiondim
        self.size = popsize
        self.cr = crossover_rate
        self.y = []
    
    def mutate(self, x1, x2, x3, x4):
        res = x1
        for i in x1:
            if np.random.random() < self.cr:
                res[i] = x2[i] + self.f * (x3 - x4)
        return res

    def update_y(self):
        #calculate function results for each
        for i in range(len(self.pop)):
            self.y.append(funct(self.pop[i]))

    def initpop(self):
        self.pop = np.random.random((self.functiondim, self.size))
        self.update_y()

    def evolve(self, ind):
        #choose four dudes
        chosen = np.random.permutation(self.size)
        #the four dudes are now vectors!
        x1 = self.pop[ind]
        x2 = self.pop[chosen(1)]
        x3 = self.pop[chosen(2)]
        x4 = self.pop[chosen(3)]
        #mutate
        mutation = self.mutate(self, x1, x2, x3, x4)
        ymutation = funct(mutation)

        if ymutation < self.y[ind]:
            self.pop[ind] = mutation
            self.y[ind] = ymutation

    def run(self, time):
        for t in range(time):
            for i in range(self.size):
                self.evolve(i) 
        

    def __str__(self):
        strvalue = ""
        for x in range(self.size):
            for y in self.pop:
                strvalue += "(" + str(y[x]) + "), "
            strvalue += "\n"
        return strvalue

pop = Populacao(1, 3, 10, 0.5)
pop.initpop()
print(pop)
pop.evolve(1)
arr = [0, 1, 0, 1]
arr = 2 * arr
print(arr)
print("\n ran!")

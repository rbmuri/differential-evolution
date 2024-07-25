import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from funct import *

chosen_function = 3

def crossover(CR, n):
    v= np.array()
    for i in range(n):
        if np.random.random() < CR:
            v.append(1)
        else:
            v.append(0)
    return v

class Populacao:
    def __init__(self, f, functiondim, popsize, crossover_rate, chosen_function):
        self.y_history = []
        self.x_history = []
        self.f = f
        self.dim = functiondim
        self.size = popsize
        self.cr = crossover_rate
        self.y = []
        self.besty = float("inf")
        self.chosen_function = chosen_function
        self.comeback_bool = 0
        self.initpop()
        
    
    def mutate(self, x1, x2, x3, x4):
        res = np.copy(x1)
        for i in range(len(x1)):
            if np.random.random() < self.cr:
                res[i] = x2[i] + self.f * (x4[i] - x3[i])
        return res

    def update_y(self):
        #calculate function results for each
        for i in range(len(self.pop)):
            output = funct(self.pop[i], self.chosen_function)
            self.y.append(output)
            if output < self.besty:
                self.besty = output
                self.bestx = self.pop[i]

    def initpop(self):
        self.pop = np.random.random((self.size, self.dim))
        self.update_y()
        self.y_history.append(self.besty)
        self.x_history.append(self.bestx)

    def evolve(self, ind):
        #choose four dudes
        chosen = np.random.permutation(self.size)
        #the four dudes are now vectors!
        x1 = self.pop[ind]
        x2 = self.pop[chosen[1]]
        x3 = self.pop[chosen[2]]
        if self.comeback_bool == 1:
            if len(self.x_history) > 4:
                if np.random.random() < 0.1:
                    x3 = self.x_history[np.random.randint(
                        len(self.x_history)-6,
                        len(self.x_history)-1
                    )]
        x4 = self.pop[chosen[3]]
        #mutate
        mutation = self.mutate(x1, x2, x3, x4)
        ymutation = funct(mutation, self.chosen_function)

        if ymutation < self.y[ind]:
            self.pop[ind] = mutation
            self.y[ind] = ymutation
            #update besty
            if ymutation < self.besty:
                self.besty = ymutation
                self.bestx = mutation

    def run(self, time):
        for t in range(time):
            for i in range(self.size):
                self.evolve(i)
            self.y_history.append(self.besty)
            self.x_history.append(self.bestx)

    def multirun(self, time):
        self.secondpop = Populacao(self.f, self.dim, self.size, self.cr)
        for t in range(time):
            for i in range(self.size):
                self.evolve(i)
            self.y_history.append(self.besty)
            self.x_history.append(self.bestx)

    def comeback(self):
        self.comeback_bool = 1    

    #prints a line for each individual (with d elements)
    def __str__(self):
        strvalue = ""
        for x in self.pop:
            for y in x:
                strvalue += "(" + str(y) + "), "
            strvalue += "\n"
        return strvalue

test1 = []
for i in range(1000):
    pop1 = Populacao(0.8, 3, 50, 0.9, 1)
    pop1.run(1000)
    test1.append(pop1.besty)

test2 = []
for i in range(100):
    pop2 = Populacao(0.8, 3, 50, 0.9, 1)
    pop2.comeback()
    pop2.run(100)
    test2.append(pop2.besty)
test1.sort(reverse=True)
test2.sort(reverse=True)

print("\nran! besty =", pop1.besty, "\n")

fig, ax = plt.subplots()

ax.plot(test1, label="Population 1")
ax.plot(test2, label="Population 2")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("besty Value")
plt.title("Differential Evolution")
plt.legend()
plt.show()


import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from funct import *
import time
import datetime

class Agente:
    def __init__(self, x, funcao):
        self.x = x
        self.y = funct(x, funcao)

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
        self.comeback_size = 5
        self.comeback_rate = 0.1
        self.initpop()
        
    def mutate(self, x1, x2, x3, x4):
        res = np.copy(x1)
        for i in range(len(x1)):
            if np.random.random() < self.cr:
                res[i] = x2[i] + self.f * (x4[i] - x3[i])

            #THIS IS WRONG, FIX IT. SHOULD BE Y VALUE
            if self.comeback_bool > 2:
                if funct(x4, self.chosen_function) < funct(x3, self.chosen_function):
                    res[i] = x2[i] + self.f * (x3[i] - x4[i])
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

    def comeback_mutation(self, x):
        if len(self.x_history) >= self.comeback_size:
            if np.random.random() < self.comeback_rate:
                x = self.x_history[np.random.randint(
                    len(self.x_history)-self.comeback_size-1,
                    len(self.x_history)-1
                )]
        return x

    def evolve(self, ind):
        #choose four dudes
        chosen = np.random.permutation(self.size)
        #the four dudes are now vectors!
        x1 = self.pop[ind]
        x2 = self.pop[chosen[1]]
        x3 = self.pop[chosen[2]]
        x4 = self.pop[chosen[3]]
        if self.comeback_bool > 0:
            x3 = self.comeback_mutation(x3)
        if self.comeback_bool > 1:
            x1 = self.comeback_mutation(x1)
            x2 = self.comeback_mutation(x2)
            x4 = self.comeback_mutation(x4)

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

    def comeback(self, size, rate, bool):
        self.comeback_bool = bool 
        self.comeback_size = size  
        self.comeback_rate = rate 

    #prints a line for each individual (with d elements)
    def __str__(self):
        strvalue = ""
        for x in self.pop:
            for y in x:
                strvalue += "(" + str(y) + "), "
            strvalue += "\n"
        return strvalue

def test(comeback_size, comeback_rate, comeback_bool):
    y_vector = []
    for i in range(2000):
        pop1 = Populacao(0.8, 3, 50, 0.9, 3)
        pop1.comeback(comeback_size, comeback_rate, comeback_bool)
        pop1.run(50)
        y_vector.append(pop1.besty)
    y_vector.sort(reverse=True)
    return y_vector

fig, ax = plt.subplots()
now = datetime.datetime.now()
print("start time:", now)
start = time.time()
ax.plot(test(5, 0, 0), label="classic - no comeback")
end = time.time()
duration = (end - start) * 3
print("end time:  ", (now + datetime.timedelta(seconds=duration)))
ax.plot(test(5, 0.1, 2), label="current best - comeback")
ax.plot(test(5, 0.1, 3), label="alternative - x4 > x3")

plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("best y value")
plt.title("Differential Evolution: Rosenbrock Funtion")
plt.legend()
plt.show()


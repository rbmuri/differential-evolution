import multiprocessing
import functools
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from funct import *
import time
import datetime

#agente, individuo, pontos, etc
class Agente:
    def __init__(self, x=None, funcao=None):
        if x is not None and funcao is not None:
            self.x = x
            self.funcao = funcao
            self.y = funct(x, funcao)
            self.ind = -1
        elif (x == -1):
            self.y = 0
        else:
            self.y = float("inf")
    
    def update(self, x):
        self.x = x
        self.y = funct(x, self.funcao)
    def update(self):
        self.y = funct(self.x, self.funcao)

    def is_equal(self, agent):
        for i in range(len(self.x)):
            if (agent.x[i] != self.x[i]):
                return False
        return True
    def __gt__(self, agent):
        if (self.y > agent.y):
            return True
        return False
    
    def copy(self, agent):
        self.x = agent.x
        self.y = agent.y

class Populacao:
    def __init__(self, f, functiondim, popsize, crossover_rate, chosen_function):
        self.pop = []
        self.history = []
        self.f = f
        self.dim = functiondim
        self.size = popsize
        self.cr = crossover_rate
        self.best = Agente()
        self.worst = Agente(-1)
        self.chosen_function = chosen_function
        self.comeback_bool = 0
        self.comeback_size = 5
        self.comeback_rate = 0.1
        self.initpop()
        
    def mutate(self, x1, x2, x3, x4):
        res = x1.x.copy()
        for i in range(len(x1.x)):
            if np.random.random() < self.cr:
                res[i] = x2.x[i] + self.f * (x4.x[i] - x3.x[i])

            if self.comeback_bool > 2:
                if x4.y < x3.y:
                    res[i] = x2.x[i] + self.f * (x3.x[i] - x4.x[i])
        mutation = Agente(res, self.chosen_function)
        mutation.ind = x1.ind
        return mutation
        

    def update_y(self):
        #calculate function results for each
        for i in range(len(self.pop_raw)):
            agent = Agente(self.pop_raw[i], self.chosen_function)
            agent.ind = len(self.pop)
            self.pop.append(agent)
            if agent.y < self.best.y:
                self.best = agent

    def initpop(self):
        self.pop_raw = np.random.random((self.size, self.dim))
        self.update_y()
        self.history.append(self.best)

    def comeback_mutation(self, x):
        if len(self.history) >= self.comeback_size:
            if np.random.random() < self.comeback_rate:
                x = self.history[np.random.randint(
                    len(self.history)-self.comeback_size-1,
                    len(self.history)-1
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

        if mutation.y < x1.y:
            self.pop[ind] = mutation
            #update besty
            if mutation.y < self.best.y:
                self.best = mutation
            if mutation.y > self.worst.y:
                self.worst = mutation
        elif x1.y > self.worst.y:
            self.worst = x1

    def run(self, time):
        for t in range(time):
            for i in range(self.size):
                self.evolve(i)
            self.history.append(self.best)

    def multirun(self, time):
        secondpop = Populacao(self.f, self.dim, (2*self.size), self.cr, self.chosen_function)
        for t in range(time):
            self.worst = Agente(-1)
            secondpop.worst = Agente(-1)
            for i in range(self.size):
                self.evolve(i)
                secondpop.evolve(i)
            if (self.best > secondpop.best):
                secondpop.pop[secondpop.worst.ind] = self.best
                secondpop.pop[secondpop.worst.ind].ind = secondpop.worst.ind
            else:
                self.pop[self.worst.ind] = secondpop.best
                self.pop[self.worst.ind].ind = self.worst.ind
                self.best = secondpop.best
            self.history.append(self.best)

    def comeback(self, size, rate, bool):
        self.comeback_bool = bool 
        self.comeback_size = size  
        self.comeback_rate = rate 

    #prints a line for each individual (with d elements)
    def __str__(self):
        strvalue = ""
        for x in self.pop:
            for y in x.x:
                strvalue += "(" + str(y) + "), "
            strvalue += "\n"
        return strvalue


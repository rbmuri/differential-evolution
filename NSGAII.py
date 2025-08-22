import multiprocessing
import functools
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from multifunct import *
import time
import math
import datetime
from pymoo.indicators.hv import HV


#must come sorted
def paretoOptimal(pop, starting_rank):
    range_n = len(pop.pop[0].y)
    lowest = [float("inf")] * range_n
    out = True
    for individual in pop.pop:
        if individual.rank >= 0:
            continue
        else:
            for i in range(len(individual.y)):
                if individual.y[i] < lowest[i]:
                    individual.rankme(starting_rank)
                    lowest[i] = individual.y[i]
                    out = False
            
    if out: return
    paretoOptimal(pop, starting_rank+1)
    return

def crowdingDistance(pop):
    sorted_dimensions = []
    var = 0
    for y in pop.pop[0].y:
        sorted_dimensions.append(sorted(pop.pop, key=lambda v: v.y[var]))
        var = var + 1
    var = 0
    for list in sorted_dimensions:
        prev = float('inf')
        next = 0
        for i in range(len(list)):
            if i == len(list) - 1:
                next = float('inf')  # No next element for the last item
            else:
                next = abs(list[i].y[var] - list[i + 1].y[var])
            list[i].crowd += next + prev
            prev = next
        var = var + 1






class Agente:
    def __init__(self, x=None, funcao=None):
        if x is None or funcao is None:
            print(x + " and " + funcao)
        if x is not None and funcao is not None:
            self.x = x
            self.funcao = funcao
            self.y = multifunct(x, funcao)
            self.ind = -1
            self.evaluated = False
            self.rank = -1
            self.crowd = -1
        elif (x == -1):
            self.y = 0
        else:
            self.y = float("inf")
    
    def update(self, x):
        self.x = x
        self.y = multifunct(x, self.funcao)
    def update(self):
        self.y = multifunct(self.x, self.funcao)


    def rankme(self, x):
        self.rank = x

    def is_equal(self, agent):
        for i in range(len(self.x)):
            if (agent.x[i] != self.x[i]):
                return False
        return True
    def __gt__(self, agent):
        if (self.rank > agent.rank):
            return True
        return False
    
    def copy(self, agent):
        self.x = agent.x
        self.y = agent.y

    def __str__(self):
        strvalue = ""
        for y in self.x:
            strvalue += "(" + str(y) + "), "
        strvalue += "\n"
        return strvalue

def generational_distance(population, fun):
    ref_front = recorded_fronts(fun)
    front = []
    all_distances = []
    for agent in population:
        if agent.rank == 0:
            front.append(agent)
    for agent in front:
        closest = float("inf")
        for ref in ref_front:
            distance = np.linalg.norm(np.array(ref) - np.array(agent.y))
            if distance < closest:
                closest = distance
        all_distances.append(closest)
    sum = 0
    for i in all_distances:
        sum = sum + i
    return (sum/len(all_distances))

def inverted_gd(population, fun):
    ref_front = recorded_fronts(fun)
    front = []
    all_distances = []
    for agent in population:
        if agent.rank == 0:
            front.append(agent)
    for agent in ref_front:
        closest = float("inf")
        for ref in front:
            distance = np.linalg.norm(np.array(ref.y) - np.array(agent))
            if distance < closest:
                closest = distance
        all_distances.append(closest)
    sum = 0
    for i in all_distances:
        sum = sum + i
    return (sum/len(all_distances))

def n_nds(pop):
    count = 0
    for i in pop:
        if i.rank == 0:
            count = count + 1
    return count

def hv(pop):
    paretofront = []
    for i in pop:
        if i.rank == 0:
            paretofront.append(np.array(i.x))
        # Reference point (must be worse than any point in F)
    ref_point = np.array([1000, 1000])
    paretofront = np.array(paretofront)
    # Initialize and calculate hypervolume
    hv = HV(ref_point=ref_point)
    hv_value = hv.do(paretofront)
    return hv_value
        

class Populacao:
    def __init__(self, f, functiondim, popsize, crossover_rate, chosen_function):
        self.pop = []
        self.mutatedpop = []
        self.rank = []
        self.history = []
        self.f = f
        self.dim = functiondim
        self.size = popsize
        self.cr = crossover_rate
        self.chosen_function = chosen_function
        self.comeback_bool = 0
        self.comeback_size = 5
        self.comeback_rate = 0.1
        self.initpop()

#    def updaterank(self):
#        popcopy = self.pop.copy()
#        rank = 0
#        while len(self.popcopy) > 0:
#            pareto = paretoOptimal(popcopy) 
#            for i in pareto:                   

    def mutate(self, x1, x2, x3, x4):
        res = x1.x.copy()
        for i in range(len(x1.x)):
            if np.random.random() < self.cr:
                res[i] = x2.x[i] + self.f * (x4.x[i] - x3.x[i])
                res[i] = search_domain(res[i], self.chosen_function)
            if self.comeback_bool > 2:
                if x4.y < x3.y:
                    res[i] = x2.x[i] + self.f * (x3.x[i] - x4.x[i])
        mutation = Agente(res, self.chosen_function)
        mutation.ind = x1.ind
        return mutation
    

    def SBX(self, p1, p2):
        xmin = 0
        nc = 10 #how far from parents the offspring will be
        c1 = []
        c2 = []
        for i in range(len(p1.x)):
            if p1.x[i] < p2.x[i]:
                x1 = p1.x[i]
                x2 = p2.x[i]
            else:
                x1 = p2.x[i]
                x2 = p1.x[i]
            beta  = 1 + (2 * (x1 - xmin)) / (x2 - x1)
            alpha = 2 - pow(beta, -(nc + 1))
            u = np.random.random()
            if u <= (1 / alpha):
                betaq = pow(u * alpha, 1 / (nc + 1))
            else:
                betaq = pow(1 / (2 - u * alpha), 1 / (nc + 1))
            c1.append( ((x1 + x2) - betaq * (x2 - x1)) * 0.5 )
            c2.append( ((x1 + x2) + betaq * (x2 - x1)) * 0.5 )


        return c1, c2
        

    def update_y(self):
        #calculate function results for each
        for i in range(len(self.pop_raw)):
            agent = Agente(self.pop_raw[i], self.chosen_function)
            agent.ind = len(self.pop)
            self.pop.append(agent)
#            if agent.y < self.best.y:
#                self.best = agent

    def constraints(self):

    def initpop(self):
        self.pop_raw = np.random.random((self.size, self.dim))
        self.update_y()

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
        if False:
            mutation, mutation2 = self.SBX(x1, x2)
            mutation = Agente(mutation, self.chosen_function)
            mutation2 = Agente(mutation2, self.chosen_function)
            self.mutatedpop.append(mutation2)
        self.mutatedpop.append(mutation)
#        if mutation.y < x1.y:
#            self.pop[ind] = mutation
#            #update besty
#            if mutation.y < self.best.y:
#                self.best = mutation
#            if mutation.y > self.worst.y:
#                self.worst = mutation
#        elif x1.y > self.worst.y:
#            self.worst = x1

    def run(self, time):
        print('''
n_gen  |   n_eval  |  n_nds |      igd      |       gd      |       hv
==========================================================================''')
        for t in range(time):
            for i in range(self.size):
                self.evolve(i)
            self.darwinism()
            try: print(f"{t+1:>6} | {n_eval():>9} | {n_nds(self.pop):>6} | {inverted_gd(self.pop, self.chosen_function):>13.10f} | {generational_distance(self.pop, self.chosen_function):>13.10f} | {hv(self.pop):>13.6E}")
            except Exception as e: print("Collecting ideal front data.")
            #self.history.append(self.best)
        

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
                

    def darwinism(self):
        self.pop = self.pop + self.mutatedpop
        for i in self.pop:
            i.rank = -1
        self.pop = sorted(self.pop, key=lambda v: tuple(v.y))
        
        paretoOptimal(self, 0)
        crowdingDistance(self)
        self.pop = sorted(self.pop, key=lambda v: (v.rank, -v.crowd))
        self.pop = self.pop[slice(self.size)]
        self.mutatedpop = []
        #self.best = self.pop[0]


    #prints a line for each individual (with d elements)
    def __str__(self):
        strvalue = ""
        for x in self.pop:
            for y in x.x:
                strvalue += "(" + str(y) + "), "
            strvalue += "\n"
        return strvalue


import numpy as np
import math
from funct import funct

class DE:

    def __init__(self):

        self.history = []
        self.best_fit = math.inf
        self.best_sol = []
        self.F_history = []
        self.Cr_history = []
        self.N1_history = []
        self.N2_history = []
    
    def unmodified_DE(self, fun = 1, N = 50, dim = 2, F = 0.9, Cr = 0.7, max_evaluations = 200, lb = -20, ub = 20): # DE with no modifications

        lower_bound = np.ones((dim, 1)) * lb
        upper_bound = np.ones((dim, 1)) * ub

        self.best_fit = math.inf
        self.history = np.zeros(max_evaluations)
        self.best_sol = np.zeros((dim, 1))

        population = lower_bound + (upper_bound - lower_bound) * np.random.rand(dim, N)

        population_fitness = np.zeros((N, 1))

        for i in range(N):
            xi = population[:, i]
            population_fitness[i] = funct(xi, fun)



        for eval in range(max_evaluations):

            new_population, new_fitness, F_dt, Cr_dt = self.pop_gen(fun, population, population_fitness, F, F, Cr, Cr)

            self.best_fit = min(self.best_fit, np.min(new_fitness))

            self.history[eval] = self.best_fit
            
            population = new_population
            population_fitness = new_fitness

    def DE_F(self, fun = 1, N = 50, dim = 2, max_evaluations = 200, lb = -20, ub = 20, 
               F_max = 0.9, F_min = 0.5, Cr = 0.7, randomize = False, prob = 0.05):
        
        self.history = np.zeros(max_evaluations)
        self.F_history = np.zeros(max_evaluations)
        self.best_fit = math.inf
        self.best_sol = np.zeros((dim, 1))

        
        lower_bound = np.ones((dim, 1)) * lb
        upper_bound = np.ones((dim, 1)) * ub


        population = lower_bound + (upper_bound - lower_bound) * np.random.rand(dim, N)

        population_fitness = np.zeros((N, 1))

        for i in range(N):
            xi = population[:, i]
            population_fitness[i] = funct(xi, fun)

        for eval in range(max_evaluations):

            new_population, new_fitness, F_data, Cr_data = self.pop_gen(fun, population, population_fitness, F_min, F_max, Cr, Cr, randomize, prob)

            self.best_fit = min(self.best_fit, np.min(new_fitness))

            improv = (population_fitness - new_fitness).reshape(N)

            improv_data = [(improv[i], F_data[i]) for i in range(N)]
            improv_data = list(filter(lambda x: x[0] > 0, improv_data))

            F_mean = 0

            if len(improv_data) > 0:

                improv_data = sorted(improv_data, reverse=True)
                improv_data = improv_data[:math.ceil(len(improv_data)/2)]

                for i in range(len(improv_data)):
                    F_mean += improv_data[i][1]
                F_mean /= len(improv_data)


                F_min = max(0.0001, F_mean - 0.1)
                F_max = min(1, F_mean + 0.1)


            self.history[eval] = self.best_fit
            self.F_history[eval] = (F_min + F_max)/2

            population = new_population
            population_fitness = new_fitness

    def DE_Cr(self, fun = 1, N = 50, dim = 2, max_evaluations = 200, lb = -20, ub = 20, 
               Cr_max = 0.9, Cr_min = 0.75, F = 0.9):
        
        self.history = np.zeros(max_evaluations)
        self.Cr_history = np.zeros(max_evaluations)
        self.best_fit = math.inf
        self.best_sol = np.zeros((dim, 1))

        
        lower_bound = np.ones((dim, 1)) * lb
        upper_bound = np.ones((dim, 1)) * ub


        population = lower_bound + (upper_bound - lower_bound) * np.random.rand(dim, N)

        population_fitness = np.zeros((N, 1))

        for i in range(N):
            xi = population[:, i]
            population_fitness[i] = funct(xi, fun)

        for eval in range(max_evaluations):

            new_population, new_fitness, F_data, Cr_data = self.pop_gen(fun, population, population_fitness, F, F, Cr_min, Cr_max)

            self.best_fit = min(self.best_fit, np.min(new_fitness))

            improv = (population_fitness - new_fitness).reshape(N)

            improv_data = [(improv[i], Cr_data[i]) for i in range(N)]
            improv_data = list(filter(lambda x: x[0] > 0, improv_data))

            Cr_mean = 0

            if len(improv_data) > 0:

                improv_data = sorted(improv_data, reverse=True)
                improv_data = improv_data[:math.ceil(len(improv_data)/2)]

                for i in range(len(improv_data)):
                    Cr_mean += improv_data[i][1]
                Cr_mean /= len(improv_data)


                Cr_min = max(0.0001, Cr_mean - 0.01)
                Cr_max = min(1, Cr_mean + 0.01)


            self.history[eval] = self.best_fit
            self.Cr_history[eval] = (Cr_min + Cr_max)/2

            population = new_population
            population_fitness = new_fitness           

    def DE_FCr(self, fun = 1, N = 50, dim = 2, max_evaluations = 200, lb = -20, ub = 20, 
               Cr_max = 0.9, Cr_min = 0.75, F_max = 0.9, F_min = 0.5):
        
        self.history = np.zeros(max_evaluations)
        self.Cr_history = np.zeros(max_evaluations)
        self.F_history = np.zeros(max_evaluations)
        self.best_fit = math.inf
        self.best_sol = np.zeros((dim, 1))

        
        lower_bound = np.ones((dim, 1)) * lb
        upper_bound = np.ones((dim, 1)) * ub


        population = lower_bound + (upper_bound - lower_bound) * np.random.rand(dim, N)

        population_fitness = np.zeros((N, 1))

        for i in range(N):
            xi = population[:, i]
            population_fitness[i] = funct(xi, fun)

        for eval in range(max_evaluations):

            new_population, new_fitness, F_data, Cr_data = self.pop_gen(fun, population, population_fitness, F_min, F_max, Cr_min, Cr_max)

            self.best_fit = min(self.best_fit, np.min(new_fitness))

            improv = (population_fitness - new_fitness).reshape(N)

            improv_data = [(improv[i], F_data[i], Cr_data[i]) for i in range(N)]
            improv_data = list(filter(lambda x: x[0] > 0, improv_data))

            F_mean = 0
            Cr_mean = 0

            if len(improv_data) > 0:

                improv_data = sorted(improv_data, reverse=True)
                improv_data = improv_data[:math.ceil(len(improv_data)/2)]

                for i in range(len(improv_data)):
                    F_mean += improv_data[i][1]
                    Cr_mean += improv_data[i][2]
                F_mean /= len(improv_data)
                Cr_mean /= len(improv_data)

                F_min = max(0.0001, F_mean - 0.01)
                F_max = min(1, F_mean + 0.01)

                Cr_min = max(0.0001, Cr_mean - 0.01)
                Cr_max = min(1, Cr_mean + 0.01)


            self.history[eval] = self.best_fit
            self.Cr_history[eval] = (Cr_min + Cr_max)/2
            self.F_history[eval] = (F_min + F_max)/2

            population = new_population
            population_fitness = new_fitness           

    def DE_PopChange(self, fun = 1, N1 = 40, N2 = 30, dim = 2, F = 0.9, Cr = 0.7, max_evaluations = 200, lb = -20, ub = 20):

        lower_bound = np.ones((dim, 1)) * lb
        upper_bound = np.ones((dim, 1)) * ub

        self.best_fit = math.inf
        self.history = np.zeros(max_evaluations)
        self.N1_history = np.zeros(max_evaluations)
        self.N2_history = np.zeros(max_evaluations)
        self.best_sol = np.zeros((dim, 1))
        best_fit1 = math.inf
        best_fit2 = math.inf

        population1 = lower_bound + (upper_bound - lower_bound) * np.random.rand(dim, N1)
        population2 = lower_bound + (upper_bound - lower_bound) * np.random.rand(dim, N2)

        population_fitness1 = np.zeros((N1, 1))
        population_fitness2 = np.zeros((N2, 1))

        for i in range(N1):
            xi = population1[:, i]
            population_fitness1[i] = funct(xi, fun)

        for i in range(N2):
            xi = population2[:, i]
            population_fitness2[i] = funct(xi, fun)

        for eval in range(max_evaluations):
            new_population1, new_fitness1, F_dt, Cr_dt = self.pop_gen(fun, population1, population_fitness1, F, F, Cr, Cr)
            new_population2, new_fitness2, F_dt, Cr_dt = self.pop_gen(fun, population2, population_fitness2, F, F, Cr, Cr)

            best_fit1 = min(best_fit1, np.min(new_fitness1))
            best_fit2 = min(best_fit2, np.min(new_fitness2))

            if best_fit1 < best_fit2:
                idx1 = np.random.permutation(N1)[:4]
                idx2 = np.random.permutation(N2)[:4]

                trial_individuals1 = np.take(new_population1, idx1, axis=1)
                trial_fit1 = np.zeros(4)

                trial_individuals2 = np.take(new_population2, idx2, axis=1)
                trial_fit2 = np.zeros(4)

                for i in range(4):
                    x1 = trial_individuals1[:, i]
                    x2 = trial_individuals2[:, i]

                    trial_fit1[i] = funct(x1, fun)
                    trial_fit2[i] = funct(x2, fun)


                new_individuals1, new_fit1, F_dt, Cr_dt = self.pop_gen(fun, trial_individuals1, trial_fit1, F, F, Cr, Cr)
                new_individuals2, new_fit2, F_dt, Cr_dt = self.pop_gen(fun, trial_individuals2, trial_fit2, F, F, Cr, Cr)

                new_population1 = np.append(new_population1, new_individuals1, axis=1)
                new_population2 = np.append(new_population2, new_individuals2, axis=1)
                new_fitness1 = np.append(new_fitness1, new_fit1)
                new_fitness2 = np.append(new_fitness2, new_fit2)

                N1 += 4
                N2 += 4

            elif N2 > 8:
                idx1 = np.random.permutation(N1)[:4]
                idx2 = np.random.permutation(N2)[:4]

                new_population1 = np.delete(new_population1, idx1, axis=1)
                new_fitness1 = np.delete(new_fitness1, idx1)

                new_population2 = np.delete(new_population2, idx2, axis=1)
                new_fitness2 = np.delete(new_fitness2, idx2)

                N1 -= 4
                N2 -= 4

            self.best_fit = min(best_fit1, best_fit2)
            self.history[eval] = self.best_fit
            self.N1_history[eval] = N1
            self.N2_history[eval] = N2

            population1 = new_population1
            population2 = new_population2
            population_fitness1 = new_fitness1
            population_fitness2 = new_fitness2

    def pop_gen(self, fun, population, population_fitness, F_min, F_max, Cr_min, Cr_max, randomize = False, prob = 0.05):

        dim, N = population.shape

        new_population = np.zeros((dim, N))
        new_fitness = np.zeros((N, 1))

        F_data = np.zeros(N)
        Cr_data = np.zeros(N)

        for i in range(N):

            F = F_min + (F_max - F_min) * np.random.rand() 
            Cr = Cr_min + (Cr_max - Cr_min) * np.random.rand() 

            if randomize:
                flag = np.random.rand()

                if flag < prob:
                    F = 0.1 + np.random.rand()

            F_data[i] = F
            Cr_data[i] = Cr

            idx = np.random.permutation(N - 1)[:3]

            candidates = [j for j in range(N)]
            del candidates[i]

            i2 = candidates[idx[0]]
            i3 = candidates[idx[1]]
            i4 = candidates[idx[2]]

            x1 = population[:, i]
            x2 = population[:, i2]
            x3 = population[:, i3]
            x4 = population[:, i4]

            A = self.crossover(dim, Cr)
            nA = 1 - A

            trial = nA * x1 + A * (x2 + F * (x3 - x4))
            trial_fitness = funct(trial.reshape(dim, 1), fun)

            if trial_fitness < population_fitness[i]:
                new_population[:, i] = trial
                new_fitness[i] = trial_fitness

            else:
                new_population[:, i] = x1
                new_fitness[i] = population_fitness[i]

        return new_population, new_fitness, F_data, Cr_data

    def crossover(self, dim, Cr):

        A = np.zeros(dim)

        for d in range(dim):
            if np.random.rand() < Cr:
                A[d] = 1
            
            else:
                A[d] = 0

        return A
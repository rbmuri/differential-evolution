import numpy as np
import math
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
import atexit
import os

problem = get_problem("zdt1")
score = 0


def multifunct(x, fun):
    global score
    score = score +1

    n = len(x)
    Q = matpi(n)

    if fun == "bnh": #bihn and korn
        y = [
         4*x[0]**2 + 4*x[1]**2 ,
         (x[0]-5)**2 + (x[1]-5)**2
        ]
        return y
        
    elif fun == "zdt1":  # Manual ZDT1
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (n - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return [f1, f2]


    if fun == 10:  # ZDT1
        problem = get_problem("zdt1")
        return problem.evaluate(x)
    elif fun == 2: # Sphere function
        problem = get_problem("zdt1")
        return problem.evaluate(x)

    elif fun == 3: # Rosenbrock function
        y = 0

        for i in range(n-1):
            y = y + 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2

        return y

    elif fun == 4: # rotated Rastrigin function
        z = Q @ x
        y = 0

        for i in range(n):
            y = y + z[i]**2 - 10 * math.cos(2 * math.pi * z[i]) + 10

        return y

    elif fun == 5: # rotated Ackley function
        z = Q @ x
        cz = np.cos(2 * math.pi * z)

        y = -20 * math.exp(-0.2 * math.sqrt(z.T @ z / n)) - math.exp(np.sum(cz)/n) + 20 + math.exp(1)

        return y
    
    elif fun == 6: # Schwefel function
        y = 0

        for i in range(n+1):
            y = y + (np.sum(x[0:i]))**2

        return y

def n_eval():
    return score


def matpi(n):

    npi = math.ceil(n**2/15)
    a = ''

    for i in range(1, npi + 1):
        # a += '{:.{}}'.format(math.pi ** i, 17)
        a += '{:.{}}'.format(math.pi ** i, 17).replace('e', '').replace('+', '').replace('-', '')


    a = a.replace('.', '')

    M = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):

            ind = i*n + j
            M[i,j] = int(a[ind])

    M = np.dot(M.T, M)

    M = M / np.linalg.norm(M, 2)

    return M

def search_domain(mutation, fun):
    if fun == "zdt1":
        mutation = np.clip(mutation, 0.0, 1.0)
        return mutation
    else:
        return mutation
    
def recorded_fronts(fun):
    file_path = os.path.join(os.path.dirname(__file__), f"{fun}_front.txt")
    data = np.loadtxt(file_path)
    return data.tolist()
    
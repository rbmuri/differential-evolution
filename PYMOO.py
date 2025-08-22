from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize

problem = get_problem("zdt1")


from pymoo.core.problem import Problem
import numpy as np


algorithm = NSGA2(pop_size=100)

termination = get_termination("n_gen", 100)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

print("Best solutions (first 5):")
print(res.F[:5])

import matplotlib.pyplot as plt

plt.scatter(res.F[:, 0], res.F[:, 1])
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("NSGA-II Result")
plt.grid(True)
plt.show()

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.termination import get_termination
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.sampling.rnd import FloatRandomSampling


class MyCustomDE(Algorithm):
    def __init__(self, pop_size=100, cr=0.9, f=0.5, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.cr = cr
        self.f = f
        self.initialization = FloatRandomSampling()

    def _initialize(self):
        self.pop = self.initialization.do(self.problem, self.pop_size)
        self.n_gen = 0

    def _next(self):
        pop = self.pop
        n = len(pop)
        X = pop.get("X")

        off = []
        for i in range(n):
            a, b, c = self._select_three(i, n)
            x = X[i]
            xr1, xr2, xr3 = X[a], X[b], X[c]

            mutant = xr1 + self.f * (xr2 - xr3)
            mutant = np.clip(mutant, self.problem.xl, self.problem.xu)

            cross_points = np.random.rand(len(x)) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(x))] = True
            trial = np.where(cross_points, mutant, x)

            ind = Individual(X=trial)
            off.append(ind)

        self.off = Population.create(*off)

    def _select_three(self, current_idx, n):
        idxs = np.arange(n)
        idxs = np.delete(idxs, current_idx)
        np.random.shuffle(idxs)
        return idxs[0], idxs[1], idxs[2]

    def _replace(self):
        pop = self.pop
        off = self.off

        pop.evaluate(self.problem)
        off.evaluate(self.problem)

        new_pop = []
        for p, o in zip(pop, off):
            if self.problem.n_obj == 1:
                new_pop.append(p if p.F < o.F else o)
            else:
                new_pop.append(p if np.all(p.F <= o.F) else o)

        self.pop = Population.create(*new_pop)

problem = get_problem("zdt1")

algorithm = MyCustomDE(pop_size=100, cr=0.9, f=0.5)

termination = get_termination("n_gen", 100)

res = minimize(
    problem,
    algorithm,
    termination=termination,
    seed=1,
    verbose=True
)

# Visualize Pareto Front
Scatter(title="Custom DE on ZDT1").add(res.F).show()

# Calculate Hypervolume
hv = Hypervolume(ref_point=np.array([1.1, 1.1]))
print("HV:", hv.do(res.F))

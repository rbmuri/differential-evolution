import numpy as np
import math

def funct(x, fun):

    n = len(x)
    Q = matpi(n)

    if fun == 1:  # Quadratic function
        phi = 1e-3
        eps = 1e-7

        v1 = np.ones((n, 1))/math.sqrt(n)

        # SVD factorization for null(v1.T)
        U, S, Vt = np.linalg.svd(v1.T)
        aux = (S >= eps).sum()
        v2 = Vt[aux:].conj().T

        V = np.concatenate((v1, v2), axis=1)

        d = np.ones(n)
        d[0] = phi
        D = np.diag(d)

        Q = V.T @ D @ V / phi

        return x.T @ Q @ x

    elif fun == 2: # Sphere function
        return x.T @ x

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


def matpi(n):

    npi = math.ceil(n**2/15)
    a = ''

    for i in range(1, npi + 1):
        a += '{:.{}}'.format(math.pi ** i, 17)


    a = a.replace('.', '')

    M = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):

            ind = i*n + j
            M[i,j] = int(a[ind])

    M = np.dot(M.T, M)

    M = M / np.linalg.norm(M, 2)

    return M
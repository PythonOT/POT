import numpy as np
from ot.gromov._gw import solve_gromov_linesearch
from time import time

n = 1000

C1 = np.random.random((n, n))
C1 = C1 + C1.T
C2 = np.random.random((n, n))
C2 = C2 + C2.T

G1 = np.ones((n, n)) / (n**2)
G2 = np.eye(n) / n

tic = time()
alpha2, _, _ = solve_gromov_linesearch(G1, G2 - G1, cost_G=0, C1=C1, C2=C2, M=0, reg=1, symmetric=False)
tac = time()

print(f'Linesearch time without symmetric assumption {tac-tic}')

tic = time()
alpha1, _, _ = solve_gromov_linesearch(G1, G2 - G1, cost_G=0, C1=C1, C2=C2, M=0, reg=1, symmetric=True)
tac = time()

print(f'Linesearch time with symmetric assumption {tac-tic}')

assert alpha1 == alpha2
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

k = 1
m = 1

#Ground Truth

def Hamiltonian(p, q):
    return (0.5 * k * q**2) + (0.5 * p**2 / m)

def rhs(t, state):
    q, p = state
    dqdt = p
    dpdt = -q
    return [dqdt, dpdt]

def solver(q0, p0, t_max=20, points=300):
    
    solution = solve_ivp(rhs, [0,t_max], [q0, p0], t_eval = np.linspace(0, t_max, points))
    return (solution.t, solution.y[0], solution.y[1])
q0 = 1.0
p0 = 0.0
solution = solver(q0, p0)
plt.plot(solution[1], solution[2])
plt.axis('equal')
plt.savefig("momentum_vs_position.png")

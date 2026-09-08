import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as anp
import torch
from torch import nn
k = 1
m = 1

def Hamiltonian(state): #Embeds the physics
    q, p = state
    return (0.5 * k * anp.square(q) + (0.5 * anp.square(p) / m)) #autgrad can differentiate autograd numpy classes


def sgf(t, state): #sgf==> sympelectic gradient function
    q, p = state
    dH_dq, dH_dp = autograd.grad(Hamiltonian)(state) # taking grad syntax grad(function)(value)
    return np.array([dH_dp, -dH_dq])

def initial_energy(): #Energy based initial condition because to restric the system to have a particular energy.
    E = np.random.uniform(0.2, 1.0, 25)
    theta = np.random.uniform(0, 2*np.pi, 25)
    q0 = np.sqrt(2*E)*np.cos(theta)
    p0 = np.sqrt(2*E)*np.sin(theta)
    return list(zip(q0, p0))

def solver(q0, p0, t_max=20, points=300): #solves(dq/dt = p/m and dp/dt = -kq)
    solution = solve_ivp(sgf, [0,t_max], [q0, p0], t_eval = np.linspace(0, t_max, points))
    return solution

def generate_data(initial_state):
    states = np.array([]).reshape(0,2)
    derivatives = np.array([]).reshape(0,2)
    #grads = autograd.grad(Hamiltonian)
    for q0, p0 in initial_state:
        solutions = solver(q0, p0)
        states = np.append(states, solutions.y.T, axis=0)
        diff = np.array([sgf(t, state) for t, state in zip (solutions.t, solutions.y.T)]) # the sgf function gets the values of the state argument from the solutions.y.T
        derivatives = np.append(derivatives, diff, axis=0)
    return states, derivatives

def gaussian_noise(state, sigma = np.sqrt(0.1)):
    return state + np.random.normal(0, sigma, state.shape)

def build_data():
    initi = initial_energy()
    split = int(0.8*len(initi))

    train_state, train_derivatives = generate_data(initi[: split])
    test_state, test_derivatives = generate_data(initi[split :])

    train_state_noise = gaussian_noise(train_state)
    test_state_noise = gaussian_noise(test_state)

    state_tensor = torch.tensor(train_state_noise, dtype=torch.float32).requires_grad_() #all the arrays created by the numpy module is float64, but the tensor module expects float32, hence the conversion.
    test_tensor = torch.tensor(test_state_noise, dtype=torch.float32).requires_grad_()
    derivative_tensor = torch.tensor(train_derivatives, dtype=torch.float32)
    test_derivative_tensor = torch.tensor(test_derivatives, dtype=torch.float32)
    return state_tensor, test_tensor, derivative_tensor, test_derivative_tensor


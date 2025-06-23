import torch

class Hamiltonian():
    def __init__(self, hamiltonian_fn, q_prime, p_prime):
        self.H = hamiltonian_fn
        self.q_prime = q_prime
        self.p_prime = p_prime
    def q_prime(self, q, p, s):
        return self.q_prime(q, p, s)

    def p_prime(self, q, p, s):
        return self.p_prime(q, p, s)

class Leapfrog():
    def __init__(self, hamiltonian, step_size):
        self.hamiltonian = hamiltonian
        self.step_size = step_size

    def step(self, q, p, s):
        p_half = p + 0.5 * self.step_size * self.hamiltonian.p_prime(q, p, s)
        q_new = q + self.step_size * self.hamiltonian.q_prime(q, p_half, s + 0.5 * self.step_size)
        p_new = p_half + 0.5 * self.step_size * self.hamiltonian.p_prime(q_new, p_half, s + self.step_size)
        return q_new, p_new

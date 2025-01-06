# %% setup


import numpy as np


# %% policy


class SoftPolicy:
    def __init__(self, tau, M):
        self.tau = tau
        self.M = M


    def pi(self, S, A, theta):
        return 1 / (1 + np.exp(- A * (S @ theta) / self.tau))


# %% setup


import numpy as np


# %% data class


class MultiStageDataset:
    def __init__(self, dat, N, T, d, index, Q = None, f_S = None, degree_q_func = 2):
        self.dat = dat
        self.N = N
        self.T = T
        self.d = d
        self.index = index
        self.iS = index['iS']
        self.iP = index['iP']
        self.iA = index['iA']
        self.iR = index['iR']
        self.Q = Q
        self.degree_q_func = degree_q_func

        if f_S:
            self.f_S = f_S
        else:
            self.f_S = self.dat[:, self.iS]


    def get_phi(self, A = None):
        if A is None:
            A = self.dat[:, [self.iA]]
        self.phi = np.hstack((self.f_S, A * self.f_S))
        return self.phi


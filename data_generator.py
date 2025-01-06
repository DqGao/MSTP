# %% setup


import numpy as np
from numpy import random as rd


# %% generate data


class DataGenerator:
    def __init__(
        self, policy, N, T, d, index, 
        Mu, Sigma, ewma_alpha, sd_state, scenario
    ):
        self.policy = policy
        self.N = N
        self.T = T
        self.d = d
        self.iS = index['iS']
        self.iP = index['iP']
        self.iA = index['iA']
        self.iR = index['iR']
        self.ewma_alpha = ewma_alpha
        self.Mu = Mu
        self.Sigma = Sigma
        self.sd_state = sd_state
        self.scenario = scenario


    def new_state(self, St_ewma, At):
        if self.scenario == 1:
            return self.new_state_nonlinear(St_ewma, At)
        elif self.scenario == 2:
            return self.new_state_linear(St_ewma, At)


    def new_reward(self, St_new, At):
        if self.scenario == 1:
            return self.new_reward_nonlinear(St_new, At)
        elif self.scenario == 2:
            return self.new_reward_linear(St_new, At)


    ## nonlinear scenario
    def new_state_nonlinear(self, St_ewma, At):
        St1 = St_ewma[:, 1]
        St2 = St_ewma[:, 2]
        St3tod = St_ewma[:, 3:]
        St1_new = 0.8 * St1 + 0.3 * At * St1 + 0.2 * St2 + rd.normal(0, self.sd_state, size=self.N)
        St2_new = 0.8 * St2 - 0.3 * At * St2 + At * np.tanh((St1 - St2)/2) + rd.normal(0, self.sd_state, size=self.N)
        St3tod_new = 0.9 * St3tod + rd.normal(0, self.sd_state, size=(self.N, self.d - 3))
        return np.hstack([
            np.ones((self.N, 1)), St1_new.reshape(-1, 1), St2_new.reshape(-1, 1), St3tod_new
        ])


    def new_reward_nonlinear(self, St_new, At):
        St1 = St_new[:, 1]
        St2 = St_new[:, 2]
        return np.log(1 + np.exp(St1 + St2)) - 0.5 * At


    ## linear scenario
    def new_state_linear(self, St_ewma, At):
        St1 = St_ewma[:, 1]
        St2 = St_ewma[:, 2]
        St3tod = St_ewma[:, 3:]
        St1_new = 0.8 * St1 + 0.3 * At * St1 + 0.1 * St2 + rd.normal(0, self.sd_state, size=self.N)
        St2_new = 0.8 * St2 + 0.2 * At * St2 + 0.2 * St1 + rd.normal(0, self.sd_state, size=self.N)
        St3tod_new = 0.9 * St3tod + rd.normal(0, self.sd_state, size=(self.N, self.d - 3))
        return np.hstack([
            np.ones((self.N, 1)), St1_new.reshape(-1, 1), St2_new.reshape(-1, 1), St3tod_new
        ])


    def new_reward_linear(self, St_new, At):
        St1 = St_new[:, 1]
        St2 = St_new[:, 2]
        return St1 + St2 - 0.6 * At


    def new_data_by_user(self, theta, seed):
        dat = np.zeros((self.N, self.T + 1, self.d + 3))

        t = 0
        S = rd.default_rng(seed).multivariate_normal(self.Mu, self.Sigma, self.N)
        S = np.hstack([np.ones((self.N, 1)), S])
        dat[:, t, self.iS] = S
        P = self.policy.pi(S, np.ones(self.N), theta)
        A = 2 * rd.binomial(1, P) - 1  ## stochastic
        dat[:, t, self.iP] = (A == 1) * P + (A == -1) * (1 - P)
        dat[:, t, self.iA] = A.copy()
        S_ewma = S.copy()

        for t in range(1, self.T + 1):
            S = self.new_state(S_ewma, A) ## new S
            S_ewma = self.ewma_alpha * S + (1 - self.ewma_alpha) * S_ewma
            ## R_t is a function of S_{t+1} and A
            dat[:, t-1, self.iR] = self.new_reward(S, A)
            P = self.policy.pi(S, 1, theta)
            A = 2 * rd.binomial(1, P) - 1
            dat[:, t, self.iS] = S
            dat[:, t, self.iP] = (A == 1) * P + (A == -1) * (1 - P)
            dat[:, t, self.iA] = A.copy()
        return dat


    def new_data_combine(self, theta, seed):
        dat = self.new_data_by_user(theta, seed)
        dat = dat[:, :-1, :].reshape(self.N * self.T, self.d + 3)
        return dat


    def new_data_by_user_hard(self, theta, seed):
        dat = np.zeros((self.N, self.T + 1, self.d + 3))

        t = 0
        S = rd.default_rng(seed).multivariate_normal(self.Mu, self.Sigma, self.N)
        S = np.hstack([np.ones((self.N, 1)), S])
        dat[:, t, self.iS] = S
        A = 2 * (S @ theta >= 0) - 1  ## deterministic
        dat[:, t, self.iP] = 1
        dat[:, t, self.iA] = A.copy()
        S_ewma = S.copy()

        for t in range(1, self.T + 1):
            S = self.new_state(S_ewma, A) ## new S
            S_ewma = self.ewma_alpha * S + (1 - self.ewma_alpha) * S_ewma
            ## R_t is a function of S_{t+1} and A
            dat[:, t-1, self.iR] = self.new_reward(S, A)
            A = 2 * (S @ theta >= 0) - 1
            dat[:, t, self.iS] = S
            dat[:, t, self.iP] = 1
            dat[:, t, self.iA] = A.copy()
        return dat


    def new_data_combine_hard(self, theta, seed):
        dat = self.new_data_by_user_hard(theta, seed)
        dat = dat[:, :-1, :].reshape(self.N * self.T, self.d + 3)
        return dat


# %%

# %% setup


import numpy as np
from dataset import MultiStageDataset
from policy import SoftPolicy


# %% value functions


class ValueFunction:
    def __init__(self, policy: SoftPolicy, h = None):
        self.policy = policy
        self.h = h
    
    
    ## average reward
    def value_func_avg(self, ds: MultiStageDataset):
        return np.sum(ds.dat[:, ds.iR]) / (ds.N * ds.T)


    ## IPW
    def value_func_ipw(self, ds: MultiStageDataset, theta: np.ndarray):
        pi_t = self.policy.pi(ds.dat[:, ds.iS], ds.dat[:, ds.iA], theta)
        
        rho_t = pi_t / ds.dat[:, ds.iP]
        for t in range(1, ds.T):
            rho_t[np.arange(t, ds.N * ds.T, ds.T)]  *= rho_t[np.arange(t-1, ds.N * ds.T, ds.T)]
        for t in range(0, ds.T):
            rho_t[np.arange(t, ds.N * ds.T, ds.T)] /= np.mean(rho_t[np.arange(t, ds.N * ds.T, ds.T)])
        
        each = rho_t * ds.dat[:, ds.iR]
        return sum(each) / (ds.N * ds.T)


    ## AIPWE for each observation
    def obs_value_func_aipw(self, ds: MultiStageDataset, theta: np.ndarray):
        Q_hat = (1 - ds.dat[:, ds.iA])/2 * ds.Q[:,0] + (1 + ds.dat[:, ds.iA])/2 * ds.Q[:,1]

        pi_a1 = self.policy.pi(ds.dat[:, ds.iS], 1, theta)
        V_hat = (1 - pi_a1) * ds.Q[:,0] + pi_a1 * ds.Q[:,1]

        pi_t = self.policy.pi(ds.dat[:, ds.iS], ds.dat[:, ds.iA], theta)

        rho_t = pi_t / ds.dat[:, ds.iP]
        for t in range(1, ds.T):
            rho_t[np.arange(t, ds.N * ds.T, ds.T)] *= rho_t[np.arange(t-1, ds.N * ds.T, ds.T)]
        for t in range(0, ds.T):
            rho_t[np.arange(t, ds.N * ds.T, ds.T)] /= np.mean(rho_t[np.arange(t, ds.N * ds.T, ds.T)])
        
        rho_t_1 = np.insert(rho_t[0:-1], 0, 1)
        rho_t_1[np.arange(0, ds.N * ds.T, ds.T)] = 1
        
        each = rho_t * (ds.dat[:, ds.iR] - Q_hat) + rho_t_1 * V_hat
        out = []
        for i in range(ds.N):
            out.append(np.sum(each[(i * ds.T):((i+1) * ds.T)], 0))

        return np.array(out)


    ## APIWE
    def value_func_aipw(self, ds: MultiStageDataset, theta: np.ndarray):
        obs_value = self.obs_value_func_aipw(ds, theta)
        return sum(obs_value) / (ds.N * ds.T)


    ## derivative of value functions


    def sgn(self, a):
        if a > 0: return 1
        else: return -1


    ## gradient
    def Psi_func(self, ds: MultiStageDataset, theta_hat: np.ndarray):
        if not self.h:
            h = 1 / np.sqrt(ds.N * ds.T)

        Psi = np.zeros((ds.N, ds.d - 1))
        for j in range(1, ds.d):
            theta = theta_hat.copy()
            if theta[0] < 0.01:
                if np.abs(theta[j]) > h:
                    theta[j] -= h / 2 * self.sgn(theta[j])
                else:
                    theta *= np.sqrt(1 - (h / 2 + np.abs(theta[j])) ** 2)
            V_new = np.zeros((ds.N, 2))
            sign_h = np.array([1, -1])
            curr_norm = np.sqrt(self.policy.M - theta[1:] @ theta[1:] + theta[j]**2)
            hh = min(h, curr_norm - np.abs(theta[j]))
            for k in range(2):
                theta_new = theta.copy()
                theta_new[j] += hh * sign_h[k]
                norm2_new = theta_new[1:] @ theta_new[1:]
                theta_new[0] = (
                    np.sqrt(self.policy.M - norm2_new) * self.sgn(theta[0]) 
                    if norm2_new < self.policy.M else 0
                )
                V_new[:,k] = self.obs_value_func_aipw(ds, theta_new)
            Psi[:,j-1] = (V_new[:,0] - V_new[:,1]) / (2 * hh)
        return Psi
        
        
    ## Hessian
    def d_Psi_func(self, ds: MultiStageDataset, theta_hat: np.ndarray):
        if not self.h:
            h = 1 / np.sqrt(ds.N * ds.T)

        d_Psi = np.zeros((ds.N, (ds.d - 1)**2))
        for j in range(1, ds.d):
            for l in range(1, ds.d):
                theta = theta_hat.copy()
                if theta[0] < 0.01:
                    if j == l:
                        if np.abs(theta[j]) > h:
                            theta[j] -= h * self.sgn(theta[j])
                        else:
                            theta *= np.sqrt(1 - (h + np.abs(theta[j])) ** 2)
                    else:
                        if np.abs(theta[j]) > h and np.abs(theta[l]) > h:
                            theta[j] -= h / 2 * self.sgn(theta[j])
                            theta[l] -= h / 2 * self.sgn(theta[l])
                        elif np.abs(theta[j]) <= h and np.abs(theta[l]) > h:
                            theta *= np.sqrt(1 - (h/2 + np.abs(theta[j])) ** 2)
                            theta[l] -= h / 2 * self.sgn(theta[l])
                        elif np.abs(theta[j]) > h and np.abs(theta[l]) <= h:
                            theta *= np.sqrt(1 - (h/2 + np.abs(theta[l])) ** 2)
                            theta[j] -= h / 2 * self.sgn(theta[j])
                        else:
                            theta *= np.sqrt(
                                1 - (h / 2 + np.abs(theta[j])) ** 2 
                                - (h / 2 + np.abs(theta[l])) ** 2
                            )
                V_new = np.zeros((ds.N, 4))
                sign_h1 = np.array([1, -1, 1, -1])
                sign_h2 = np.array([1, 1, -1, -1])
                if j == l:
                    curr_norm = np.sqrt(self.policy.M - theta[1:] @ theta[1:] + theta[j]**2)
                    hh = min(h, (curr_norm - np.abs(theta[j])) / 2)
                else:
                    curr_norm1 = np.sqrt(self.policy.M - theta[1:] @ theta[1:] + theta[j]**2)
                    curr_norm2 = np.sqrt(self.policy.M - theta[1:] @ theta[1:] + theta[l]**2)
                    hh = min(
                        h, 
                        curr_norm1 - np.abs(theta[j]), 
                        curr_norm2 - np.abs(theta[l])
                    )
                for k in range(4):
                    theta_new = theta.copy()
                    theta_new[j] += hh * sign_h1[k]
                    theta_new[l] += hh * sign_h2[k]
                    norm2_new = theta_new[1:] @ theta_new[1:]
                    theta_new[0] = (
                        np.sqrt(self.policy.M - norm2_new) * self.sgn(theta[0]) 
                        if norm2_new < self.policy.M else 0
                    )
                    V_new[:,k] = self.obs_value_func_aipw(ds, theta_new)
                d_Psi[:, j-1 + (l - 1)*(ds.d - 1)] = (
                    V_new[:,0] - V_new[:,1] - V_new[:,2] + V_new[:,3]
                ) / ((2 * hh) ** 2)
        return d_Psi


# %%

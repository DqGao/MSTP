# %% setup


import numpy as np
from numpy import random as rd
from numpy import linalg as LA
from sklearn.linear_model import Lasso, LinearRegression
from scipy.optimize import minimize, NonlinearConstraint
import sys
import cvxpy as cp

from dataset import MultiStageDataset
from value_function import ValueFunction


# %% optimization


class Optimizer:
    def __init__(self, vf: ValueFunction, ncv: int):
        self.vf = vf
        self.ncv = ncv
        self.niters = 20
        self.threshold = 0.1


    def l2_constraint(self, bound, cardinality: int):
        def cons_f(x):
            return x @ x
        def cons_J(x):
            return 2 * x
        def cons_H(x, v):
            return v[0] * 2 * np.eye(cardinality)
        nonlinear_constraint = NonlinearConstraint(
            cons_f, -np.inf, bound, jac=cons_J, hess=cons_H
        )
        return nonlinear_constraint


    def soft_threshold(self, rho: float, lam: float):
        if rho < - lam:
            return (rho + lam)
        elif rho > lam:
            return (rho - lam)
        else: 
            return 0


    ## beta
        
    
    def get_Q(self, ds: MultiStageDataset, X: np.ndarray, Y: np.ndarray, lam_beta: float):
        mod_beta = Lasso(alpha=lam_beta, fit_intercept=False)
        mod_beta.fit(X, Y)
        beta = mod_beta.coef_
        pos_idx = np.nonzero(beta)[0]
        ## re-estimate on the support
        if len(pos_idx) > 0:
            mod_beta2 = LinearRegression(fit_intercept=False)
            mod_beta2.fit(X[:, pos_idx], Y)
            beta[pos_idx] = mod_beta2.coef_
        
        Q_a = np.vstack([
            ds.get_phi(A=-1) @ beta, ds.get_phi(A=1) @ beta
        ]).T
        return Q_a, beta


    ## coordinate descent


    def coordinate_descent_lasso(
        self, ds: MultiStageDataset, theta_init: np.ndarray, lam: float
    ):
        theta_curr = theta_init.copy()
        ## multiple starting points
        for j in range(ds.d):
            nudge = [-0.8, -0.4, 0, 0.4, 0.8]
            start_values = []
            for k in nudge:
                theta_new = theta_init.copy()
                theta_new[j] += k
                start_values.append(
                    - self.vf.value_func_aipw(ds, theta_new)
                )
            theta_curr[j] += nudge[np.argmin(start_values)]
        if LA.norm(theta_curr, 2) > 0:
            theta_curr = theta_curr / LA.norm(theta_curr, 2)
        else:
            theta_curr = np.zeros(ds.d)

        ## coordinate descent
        theta_old = theta_init.copy() + 1
        values = []
        thetas = []
        i = 0
        while i < self.niters and LA.norm(theta_curr - theta_old, 2) > self.threshold: 
            theta_old = theta_curr.copy()
            for j in range(ds.d):
                def f0_theta(thetaj):
                    theta_new = theta_curr.copy()
                    theta_new[j] = thetaj
                    return - self.vf.value_func_aipw(ds, theta_new)
                mod_theta = minimize(f0_theta, x0 = theta_curr[j], method='BFGS')
                theta_curr[j] = self.soft_threshold(mod_theta.x, lam)
                if LA.norm(theta_curr, 2) > 0:
                    theta_curr = theta_curr / LA.norm(theta_curr, 2)
                else:
                    theta_curr = np.zeros(ds.d)
            values.append(- self.vf.value_func_aipw(ds, theta_curr))
            thetas.append(theta_curr.copy())
            i += 1
        return theta_curr


    ## one-step estimator


    ## Dantzig estimator
    def get_w(self, d: int, H_tg: np.ndarray, H_gg: np.ndarray, lam: float):
        w = cp.Variable(shape=d - 2)
        obj = cp.Minimize(cp.norm(w, 1))
        constraints = [cp.norm(H_tg - w.T @ H_gg, "inf") <= lam]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS)
        return w.value


    ## index of theta and gamma
    def get_idx_t_g(self, d: int, j: int):
        idx_t = j - 1
        idx_g = np.delete(np.arange(d), [0, j]) - 1
        return idx_t, idx_g
        

    ## find theta tilde
    def get_theta_tilde(
        self, theta_hat: np.ndarray, d: int, j: int, 
        J: np.ndarray, H: np.ndarray, lam: float
    ):
        idx_t, idx_g = self.get_idx_t_g(d, j)
        w_hat = self.get_tilde_w(d, j, H, lam)
        S_hat = J[idx_t] - w_hat.T @ J[idx_g]
        I_tg = H[idx_t, idx_t] - w_hat.T @ H[idx_g, idx_t]
        thetaj_tilde = theta_hat[j] - S_hat / I_tg
        return thetaj_tilde


    ## find w
    def get_tilde_w(self, d: int, j: int, H: np.ndarray, lam: float):
        idx_t, idx_g = self.get_idx_t_g(d, j)
        H_tg = H[idx_t, idx_g]
        H_gg = H[np.ix_(idx_g, idx_g)]
        w_hat = self.get_w(d, H_tg, H_gg, lam)
        return w_hat


    ## get the projection error
    def get_tilde_proj_error(self, d: int, j: int, H: np.ndarray, w: np.ndarray):
        idx_t, idx_g = self.get_idx_t_g(d, j)
        H_tg = H[idx_t, idx_g]
        H_gg = H[np.ix_(idx_g, idx_g)]
        return LA.norm(H_tg - w.T @ H_gg, 1)


    ## cross validation


    def split_data(self, ds: MultiStageDataset, fold_idx: np.ndarray, k: int):
        n_train = sum(fold_idx != k)
        n_valid = sum(fold_idx == k)
        dat_train = np.vstack([
            ds.dat[(i * ds.T):((i+1) * ds.T), :] 
            for i in range(ds.N)
            if fold_idx[i] != k
        ])
        Q_train = np.vstack([
            ds.Q[(i * ds.T):((i+1) * ds.T), :] 
            for i in range(ds.N)
            if fold_idx[i] != k
        ])
        dat_valid = np.vstack([
            ds.dat[(i * ds.T):((i+1) * ds.T), :] 
            for i in range(ds.N)
            if fold_idx[i] == k
        ])
        Q_valid = np.vstack([
            ds.Q[(i * ds.T):((i+1) * ds.T), :] 
            for i in range(ds.N)
            if fold_idx[i] == k
        ])
        ds_train = MultiStageDataset(
            dat_train, n_train, ds.T, ds.d, ds.index, Q_train
        )
        ds_valid = MultiStageDataset(
            dat_valid, n_valid, ds.T, ds.d, ds.index, Q_valid
        )
        return ds_train, ds_valid


    ## tuning parameter for theta check or theta hat
    def get_lam_sparse(self, ds: MultiStageDataset, theta: np.ndarray, lam_list: list):
        v_hat = np.zeros((self.ncv, len(lam_list)))
        fold_idx = rd.choice(self.ncv, ds.N, replace=True)
        for k in range(self.ncv):
            ds_train, ds_valid = self.split_data(ds, fold_idx, k)
            for l in range(len(lam_list)):
                lam = lam_list[l]
                theta_est = self.coordinate_descent_lasso(
                    ds_train, theta, lam
                )
                v_hat[k, l] = self.vf.value_func_aipw(ds_valid, theta_est)
        v_hat_mean = np.mean(v_hat, 0)
        lam_lasso = lam_list[np.argmax(v_hat_mean)]
        return lam_lasso


    ## tuning parameter for theta tilde
    def get_lam_ose(self, ds: MultiStageDataset, j: int, H: np.ndarray, lam_list: list):
        v_hat = np.zeros((self.ncv, len(lam_list)))
        fold_idx = rd.choice(self.ncv, ds.N, replace=True)
        for k in range(self.ncv):
            H_tra = np.vstack([H[i, :] for i in range(ds.N) if fold_idx[i] != k])
            H_val = np.vstack([H[i, :] for i in range(ds.N) if fold_idx[i] == k])
            H_tra = np.mean(H_tra, 0).reshape(ds.d - 1, ds.d - 1)
            H_val = np.mean(H_val, 0).reshape(ds.d - 1, ds.d - 1)
            for l in range(len(lam_list)):
                lam = lam_list[l]
                w = self.get_tilde_w(ds.d, j, H_tra, lam)
                proj_error = self.get_tilde_proj_error(ds.d, j, H_val, w)
                v_hat[k, l] = - proj_error
        v_hat_mean = np.mean(v_hat, 0)
        lam_tilde = lam_list[np.argmax(v_hat_mean)]
        return lam_tilde


    ## tuning parameter for Q functions
    def get_lam_beta(
        self, ds: MultiStageDataset, X: np.ndarray, Y: np.ndarray, 
        theta: np.ndarray, lam_list: list
    ):
        v_hat = np.zeros((self.ncv, len(lam_list)))
        fold_idx = rd.choice(self.ncv, ds.N, replace=True)
        for k in range(self.ncv):
            ds_train, ds_valid = self.split_data(ds, fold_idx, k)
            X_train = np.vstack([
                X[(i * ds.T):((i+1) * ds.T), :] 
                for i in range(ds.N)
                if fold_idx[i] != k
            ])
            Y_train = np.vstack([
                Y[(i * ds.T):((i+1) * ds.T)] 
                for i in range(ds.N)
                if fold_idx[i] != k
            ])
            X_test = np.vstack([
                X[(i * ds.T):((i+1) * ds.T), :] 
                for i in range(ds.N)
                if fold_idx[i] == k
            ])
            Y_test = np.vstack([
                Y[(i * ds.T):((i+1) * ds.T)] 
                for i in range(ds.N)
                if fold_idx[i] == k
            ])
            for l in range(len(lam_list)):
                lam = lam_list[l]
                Q_train, beta = self.get_Q(ds_train, X_train, Y_train, lam)
                v_hat[k, l] = - np.mean((Y_test - X_test @ beta)**2)
        v_hat_mean = np.mean(v_hat, 0)
        lam_beta = lam_list[np.argmax(v_hat_mean)]
        return lam_beta


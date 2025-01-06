# %% setup


import numpy as np
from numpy import random as rd
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import pandas as pd

from dataset import MultiStageDataset
from policy import SoftPolicy
from value_function import ValueFunction
from optimizer import Optimizer

from pathlib import Path
import time
import sys
np.set_printoptions(suppress=True)
jobid = int(sys.argv[1])


# %% parameters


seed_list = 2024 * np.arange(100)
seed = seed_list[jobid]


## tuning parameters
lam_check_list = [0.05, 0.1, 0.15]
lam_hat_list = [0.05, 0.1, 0.15]
lam_tilde_list = [4, 6, 8, 10, 12]
lam_beta_list = [0.1, 0.2, 0.5, 1, 2]
ncv = 4   ## number of folds in cross validation

## features of policy
tau = 0.2
M = 1 ## upper bound on the norm of theta

## replications
conf_level = 0.95
nbs = 100
n_jobs = 20

## save to file
ver = '1'
path = 'res/'
version = f'ohio_{ver}'
filename = path + version + '.txt'
file_train = "data_train2018.txt"
file_test = "data_test2018.txt"


# %% data


rd.seed(seed)


def func_reward(x):
    if x < 80: return - (80 - x)**2 / 30
    elif 80 <= x <= 140: return 0
    else: return - (x - 140)**1.35 / 30


state_cols = [
    'basal', 'glucose_level', 'meal', 'sleep', 'work', 
    'hypo_event', 'exercise',
    'basis_heart_rate', 'basis_gsr', 'basis_skin_temperature',
    'basis_air_temperature', 'basis_steps',
]

dat_df = pd.read_csv(file_train, header=0, index_col=False)
dat_df['time'] = pd.to_datetime(dat_df['time'])
dat_df.sort_values(by=['pid', 'time'], inplace=True)
## missing data in A=bolus has been filled with 0
dat_df = dat_df.groupby('pid').apply(
    lambda group: group.interpolate(method='linear'), include_groups=False
)
dat_df['R'] = dat_df['glucose_level'].apply(func_reward)
dat_df['R'] = dat_df.groupby('pid')['R'].shift(periods=-1, fill_value=np.nan)

## states
X = dat_df[state_cols].to_numpy()
X_mean = np.nanmean(X, axis=0)
X_std = np.nanstd(X, axis=0)
X = (X - X_mean) / X_std
## action
A = 2 * (dat_df['bolus'] > 0).to_numpy() - 1
## reward
R = dat_df['R'].to_numpy()
R_mean = np.nanmean(R)
R_std = np.nanstd(R)
R = (R - R_mean) / R_std


idx_comp = ~np.isnan(X).any(axis=1)
X_prop = X[idx_comp]
A_prop = A[idx_comp]
prop_model = LogisticRegression(
    penalty='l1', C=0.001, random_state=seed, solver='liblinear'
)
prop_model.fit(X_prop, A_prop)
print(prop_model.classes_)
print(prop_model.intercept_, prop_model.coef_)
pred = prop_model.predict_proba(X_prop)
pred = pred[:, 0] * (A_prop == -1) + pred[:, 1] * (A_prop == 1)
P = np.zeros(len(A)) * np.nan
P[idx_comp] = pred


dat_all = np.hstack([
    np.ones((len(R), 1)), 
    X, 
    np.array(A).reshape(-1, 1), 
    np.array(R).reshape(-1, 1), 
    np.array(P).reshape(-1, 1)
])

## features of data
d = X.shape[1] + 1 ## intercept
T = 12
block = 4*60//5

dat_epi = dat_all.reshape(-1, block, d + 3)
dat_epi = dat_epi[:, :T, :]
idx_comp = ~np.isnan(dat_epi).any(axis=(1,2))
dat_epi = dat_epi[idx_comp]

## train
N = dat_epi.shape[0]
dat = dat_epi.reshape(-1, d + 3)
dat_by_user = dat.reshape(N, T, d + 3)


# %% test data


dat_df_test = pd.read_csv(file_test, header=0, index_col=False)
dat_df_test['time'] = pd.to_datetime(dat_df_test['time'])
dat_df_test.sort_values(by=['pid', 'time'], inplace=True)
## missing data in A=bolus has been filled with 0
dat_df_test = dat_df_test.groupby('pid').apply(
    lambda group: group.interpolate(method='linear'), include_groups=False
)
dat_df_test['R'] = dat_df_test['glucose_level'].apply(func_reward)
dat_df_test['R'] = dat_df_test.groupby('pid')['R'].shift(periods=-1, fill_value=np.nan)

## states
X_test = dat_df_test[state_cols].to_numpy()
X_test = (X_test - X_mean) / X_std
## action
A_test = 2 * (dat_df_test['bolus'] > 0).to_numpy() - 1
## reward
R_test = dat_df_test['R'].to_numpy()
R_test = (R_test - R_mean) / R_std


idx_comp_test = ~np.isnan(X_test).any(axis=1)
X_prop_test = X_test[idx_comp_test]
A_prop_test = A_test[idx_comp_test]
pred_test = prop_model.predict_proba(X_prop_test)
pred_test = pred_test[:, 0] * (A_prop_test == -1) + pred_test[:, 1] * (A_prop_test == 1)
P_test = np.zeros(len(A_test)) * np.nan
P_test[idx_comp_test] = pred_test


dat_all_test = np.hstack([
    np.ones((len(R_test), 1)), 
    X_test, 
    np.array(A_test).reshape(-1, 1), 
    np.array(R_test).reshape(-1, 1), 
    np.array(P_test).reshape(-1, 1)
])

block = 4*60//5

dat_epi_test = dat_all_test.reshape(-1, block, d + 3)
dat_epi_test = dat_epi_test[:, :T, :]
idx_comp_test = ~np.isnan(dat_epi_test).any(axis=(1,2))
dat_epi_test = dat_epi_test[idx_comp_test]

N_test = dat_epi_test.shape[0]
T_test = T
dat_test = dat_epi_test.reshape(-1, d + 3)


# %% simulation starts


Path(path).mkdir(parents=True, exist_ok=True)
if seed == 0:
    with open(filename, 'w') as f:
        f.write("")

index = {
    "iS": np.arange(d),
    "iA": d,
    "iR": d + 1,
    "iP": d + 2,
}
mstp = SoftPolicy(tau, M) ## policy
vf = ValueFunction(mstp) ## value function
opt = Optimizer(vf, ncv) ## optimizer


start = time.time()
Q_0 = np.zeros((N * T, 2)) ## initialize the Q-function vector
ds = MultiStageDataset(dat, N, T, d, index, Q_0)
value_b = vf.value_func_avg(ds)
    
    
rd.seed(seed)

##### theta_check #####
theta_init = np.zeros(d)
lam_check = opt.get_lam_sparse(ds, theta_init, lam_check_list)
theta_check_lasso = opt.coordinate_descent_lasso(
    ds, theta_init, lam_check
)
pS_check = np.nonzero(theta_check_lasso)[0]

## re-estimate on the support
def f0_theta(theta_nonzero):
    theta = np.zeros(d)
    theta[pS_check] = theta_nonzero
    return - vf.value_func_ipw(ds, theta)

mod_theta0 = minimize(
    f0_theta, x0 = theta_init[pS_check], method='trust-constr', 
    constraints=opt.l2_constraint(M, len(pS_check))
)
theta_check = np.zeros(d)
theta_check[pS_check] = mod_theta0.x

## test value
ds_test = MultiStageDataset(dat_test, N_test, T_test, d, index)
value_check = vf.value_func_ipw(ds_test, theta_check)


##### Q-function #####
Q_a = Q_0.copy()
V_all = np.zeros(N * T)
for t in range(T - 1, -1, -1):
    idx_t = np.arange(t, N * T, T)
    dat_t = ds.dat[idx_t, :]
    ds_t = MultiStageDataset(dat_t, N, 1, d, index, Q_0[idx_t, :])
    X_t = ds_t.get_phi()
    if t == T - 1:
        Y_t = dat_t[:, index['iR']]
    else:
        Y_t = dat_t[:, index['iR']] + V_all[np.arange(t + 1, N * T, T)]
    lam_beta = opt.get_lam_beta(ds_t, X_t, Y_t, theta_check, lam_beta_list)
    Q_a_t, beta = opt.get_Q(ds_t, X_t, Y_t, lam_beta)
    Q_a[idx_t, :] = Q_a_t.copy()
    pi_a1_t = mstp.pi(dat_t[:, ds.iS], 1, theta_check) ## pi(A = 1 | S)
    V_all[idx_t] = (1 - pi_a1_t) * Q_a_t[:, 0] + pi_a1_t * Q_a_t[:, 1]
## update the Q-function
ds.Q = Q_a.copy()


##### theta_hat #####
lam_hat = opt.get_lam_sparse(ds, theta_check, lam_hat_list)
theta_hat_lasso = opt.coordinate_descent_lasso(
    ds, theta_check, lam_hat
)
pS_hat = np.nonzero(theta_hat_lasso)[0]

## re-estimate on the support
def f0_theta(theta_nonzero):
    theta = np.zeros(d)
    theta[pS_hat] = theta_nonzero
    return - vf.value_func_aipw(ds, theta)

mod_theta = minimize(
    f0_theta, x0 = theta_check[pS_hat], method='trust-constr', 
    constraints=opt.l2_constraint(M, len(pS_hat))
)
theta_hat = np.zeros(d)
theta_hat[pS_hat] = mod_theta.x

## test value
ds_test = MultiStageDataset(dat_test, N_test, T_test, d, index)
value_hat = vf.value_func_ipw(ds_test, theta_hat)


##### theta_tilde #####
J = - vf.Psi_func(ds, theta_hat)
H = - vf.d_Psi_func(ds, theta_hat)

lam_tilde = np.zeros(d)
for j in range(1, d):
    lam_tilde[j] = opt.get_lam_ose(ds, j, H, lam_tilde_list)

J = np.mean(J, 0)
H = np.mean(H, 0).reshape(d-1, d-1)
theta_tilde = np.zeros(d)
theta_tilde[0] = theta_hat[0]
for j in range(1, d):
    theta_tilde[j] = opt.get_theta_tilde(theta_hat, d, j, J, H, lam_tilde[j])

## test value
ds_test = MultiStageDataset(dat_test, N_test, T_test, d, index)
value_tilde = vf.value_func_ipw(ds_test, theta_tilde)


seeds_bs = np.random.randint(0, 2**32, size=nbs)
def bootstrapping(b):
    seed_bs = seeds_bs[b]
    rd.seed(seed_bs)
    id_bs = rd.choice(N, N, replace=True)
    dat_bs = np.vstack([dat_by_user[i][0:T] for i in id_bs])
    Q_bs = np.vstack([Q_a[(i*T):((i+1)*T), :] for i in id_bs])
    ds_bs = MultiStageDataset(dat_bs, N, T, d, index, Q_bs)
    ##### theta_hat #####
    theta_bs_hat_lasso = opt.coordinate_descent_lasso(
        # start from the original theta_check
        ds_bs, theta_check, lam_hat
    )
    pS_bs_hat = np.nonzero(theta_bs_hat_lasso)[0]
    
    ## re-estimate on the support
    def f0_theta(theta_nonzero):
        theta = np.zeros(d)
        theta[pS_bs_hat] = theta_nonzero
        return - vf.value_func_aipw(ds_bs, theta)

    mod_theta = minimize(
        f0_theta, x0 = theta_check[pS_bs_hat], method='trust-constr', 
        constraints=opt.l2_constraint(M, len(pS_bs_hat))
    )
    theta_bs_hat = np.zeros(d)
    theta_bs_hat[pS_bs_hat] = mod_theta.x
    
    ##### theta_tilde #####
    J_bs = - vf.Psi_func(ds_bs, theta_bs_hat)
    J_bs = np.mean(J_bs, 0)
    H_bs = - vf.d_Psi_func(ds_bs, theta_bs_hat)
    H_bs = np.mean(H_bs, 0).reshape(d-1, d-1)
    
    theta_bs_tilde = np.zeros(d)
    theta_bs_tilde[0] = theta_bs_hat[0]
    for j in range(1, d):
        theta_bs_tilde[j] = opt.get_theta_tilde(theta_bs_hat, d, j, J_bs, H_bs, lam_tilde[j])
    return [theta_bs_hat, theta_bs_tilde]

results = Parallel(n_jobs=n_jobs)(
    delayed(bootstrapping)(b) for b in range(nbs)
)
results = np.array(list(results))

boots_hat = results[:, 0, :]
boots_tilde = results[:, 1, :]


## CI and MAD
mad_hat = np.zeros(d)
CI_hat = np.zeros((d, 2))
for j in range(d):
    boots_j = boots_hat[:,j]
    mad_hat[j] = median_abs_deviation(boots_j)
    CI_hat[j, 0] = np.quantile(boots_j, (1 - conf_level) / 2)
    CI_hat[j, 1] = np.quantile(boots_j, (1 + conf_level) / 2)

mad_tilde = np.zeros(d)
CI_tilde = np.zeros((d, 2))
for j in range(d):
    boots_j = boots_tilde[:,j]
    mad_tilde[j] = median_abs_deviation(boots_j)
    CI_tilde[j, 0] = np.quantile(boots_j, (1 - conf_level) / 2)
    CI_tilde[j, 1] = np.quantile(boots_j, (1 + conf_level) / 2)


out = np.hstack([
    np.array([
        value_b, value_check, value_hat, value_tilde,
        seed, lam_check, lam_hat, lam_beta, 
    ]), 
    lam_tilde, 
    theta_check, 
    theta_hat, mad_hat, CI_hat.reshape(2*d), 
    theta_tilde, mad_tilde, CI_tilde.reshape(2*d), 
])

with open(filename, 'a') as f:
    np.savetxt(f, out.reshape((1, -1)), fmt='%.4f')


# %%

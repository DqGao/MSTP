# %% setup

import numpy as np
from numpy import random as rd

from dataset import MultiStageDataset
from policy import SoftPolicy
from value_function import ValueFunction
from data_generator import DataGenerator

from pathlib import Path
import sys
np.set_printoptions(suppress=True)
jobid = int(sys.argv[1])


# %% data model

T_list = [1, 5, 10]
theta1_start_list = np.arange(-1, 1, 0.1)
array_idx = np.unravel_index(jobid, (len(T_list), len(theta1_start_list)))
T = T_list[array_idx[0]]
theta1_start = theta1_start_list[array_idx[1]]
if theta1_start == 0.9:
    theta1_list = np.arange(theta1_start, theta1_start + 0.11, 0.01)
else:
    theta1_list = np.arange(theta1_start, theta1_start + 0.1, 0.01)
theta2_list = np.arange(-1, 1.01, 0.01)


## features of data
d = 3
N_test = 200000
T_test = T

## features of data generator
ewma_alpha = 0.8
Mu = np.zeros(d - 1)
Sigma = (
    4 * np.eye(d - 1) + 
    0.2 * np.diag(np.ones(d - 2), k=1) + 
    0.2 * np.diag(np.ones(d - 2), k=-1)
)
Sigma[0, 1] = Sigma[1, 0] = 1
sd_state = 0.4
theta_b = np.zeros(d)

## features of policy
tau = 0.2
M = 1 ## upper bound on the norm of theta

## save to file
ver = '1'
path = 'opt' + ver + '/'
version = f'True{ver}_soft_T{T_test}_N{N_test}_d{d-1}_tau{tau}'
filename = path + version + '.txt'


# %% simulation starts


Path(path).mkdir(parents=True, exist_ok=True)
if theta1_start == -1:
    with open(filename, 'w') as f:
        f.write("")


# def true_value(theta1):
index = {
    "iS": np.arange(d),
    "iA": d,
    "iR": d + 1,
    "iP": d + 2,
}
mstp = SoftPolicy(tau, M)
vf = ValueFunction(mstp)
generator_test = DataGenerator(
    mstp, N_test, T_test, d, index, 
    Mu, Sigma, ewma_alpha, sd_state, scenario=1
)

out = []
for theta1 in theta1_list:
    seed = int(200 * (theta1 + 1))
    rd.seed(seed)
    ## negative theta0
    seeds_theta = np.random.randint(0, 2**32, size=len(theta2_list))
    for i, theta2 in enumerate(theta2_list):
        if theta1**2 + theta2**2 > M:
            out.append([np.nan, theta1, theta2, np.nan])
        else:
            theta0 = (
                - np.sqrt(M - theta1**2 - theta2**2)
                if theta1**2 + theta2**2 < M
                else 0
            )
            theta = np.array([theta0, theta1, theta2])
            dat_test = generator_test.new_data_combine(theta, seeds_theta[i])
            ds_test = MultiStageDataset(dat_test, N_test, T_test, d, index)
            value_test = vf.value_func_avg(ds_test)
            out.append([theta0, theta1, theta2, value_test])

    ## positive theta0
    seeds_theta = np.random.randint(0, 2**32, size=len(theta2_list))
    for theta2 in theta2_list:
        if theta1**2 + theta2**2 > M:
            out.append([np.nan, theta1, theta2, np.nan])
        else:
            theta0 = (
                np.sqrt(M - theta1**2 - theta2**2)
                if theta1**2 + theta2**2 < M
                else 0
            )
            theta = np.array([theta0, theta1, theta2])
            dat_test = generator_test.new_data_combine(theta, seeds_theta[i])
            ds_test = MultiStageDataset(dat_test, N_test, T_test, d, index)
            value_test = vf.value_func_avg(ds_test)
            out.append([theta0, theta1, theta2, value_test])


with open(filename, 'a') as f:
    np.savetxt(f, np.array(out), fmt='%.3f')


# Asymptotic Inference for Multi-Stage Stationary Treatment Policy with Variable Selection

- **Main scripts for simulations**
  - `MSTP0.py`, `MSTP1.py`, and `MSTP2.py` run experiments for $Q^{(0)}$, $Q^{(1)}$, and $Q^{(2)}$, respectively.

- **Helper functions**
  - `dataset.py`: Creates a container to store the data.
  - `optimizer.py`: Provides functions to estimate $\check{\theta}$, $\hat{\theta}$, and $\tilde{\theta}$.
  - `policy.py`: Defines the MSTP class.
  - `value_function.py`: Estimates the value function, its gradient, and the Hessian matrix.

- **True optimal parameter**
  - `TrueOptimal.py`: Finds the true optimal parameters via grid search.

- **Job submission**
  - `run_res.sh`: Submits the experiments `MSTP0.py`, `MSTP1.py`, or `MSTP2.py`.
  - `run_opt.sh`: Submits `TrueOptimal.py`.

- **Real data analysis**
  - `T1DM_preproc.py`: Preprocesses the dataset, with training and test data cleaned separately.
  - `T1DM.py`: Runs the experiments on the real data.
  - `run_t1dm.py`: Submits the experiment `T1DM.py`.

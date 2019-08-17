#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_bl_equilibrium_ret [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bl_equilibrium_ret&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-blreturns).

# +
import numpy as np
import pandas as pd

from arpym.estimation import cov_2_corr, exp_decay_fp
from arpym.statistics import meancov_sp
from arpym.views import black_litterman, min_rel_entropy_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-parameters)

c = 0.5  # confidence level in the views
c_uninf = 1e-6  # confidence level in the uninformative views
eta = np.array([1, -1])  # parameters for qualitative views
lam = 1.2  # average risk-aversion level
tau_hl = 1386  # half life parameter
v = np.array([[1, - 1, 0], [0, 0, 1]])  # pick matrix
w = np.array([1/3, 1/3, 1/3])  # market-weighted portfolio

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step00): Load data

path = '../../../databases/global-databases/equities/db_stocks_SP500/'
data = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step01): Compute time series of returns

n_ = len(w)  # market dimension
r_t = data.pct_change().iloc[1:, :n_].values  # returns

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step02): Compute the sample mean and the exponential decay sample covariance

t_ = len(r_t)
p_t_tau_hl = exp_decay_fp(t_, tau_hl)  # exponential decay probabilities
mu_hat_r, sig2_hat_r = meancov_sp(r_t, p_t_tau_hl)  # sample mean and cov.

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step03): Compute Black-Litterman prior parameters

# +
# expectation in terms of market equilibrium
mu_r_equil = 2 * lam * sig2_hat_r @ w

tau = t_  # uncertainty level in the reference model
mu_m_pri = mu_r_equil
cv_pri_pred = (1 + 1 / tau) * sig2_hat_r
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step04): Compute vectors quantifying the views

i = v @ mu_r_equil + eta * np.sqrt(np.diag(v @ sig2_hat_r@ v.T))
sig2_i_mu = ((1 - c) / (tau * c)) * (v @ sig2_hat_r @ v.T)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step05): Compute effective rank corresponding to the pick matrix

# +

def eff_rank(s2):
    lam2_n, _ = np.linalg.eig(s2)
    wn = lam2_n / np.sum(lam2_n)
    return np.exp(- wn @ np.log(wn))


eff_rank = eff_rank(cov_2_corr(v @ sig2_hat_r @ v.T)[0])
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step06): Compute Black-Litterman posterior parameters

mu_m_pos, cv_pos_pred = black_litterman(mu_r_equil, sig2_hat_r, tau, v, i,
                                        sig2_i_mu)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step07):  Compute Black-Litterman posterior parameters in the case of uninformative views

# +
# compute vector quantifying the views in covariance
sig2_unifview = ((1 - c_uninf) / c_uninf) * v @ sig2_hat_r @ v.T

mu_m_pos, cv_pos_pred = black_litterman(mu_r_equil, sig2_hat_r, tau, v,
                                        i, sig2_unifview)
# -

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step08): Compute Black-Litterman sure posterior parameters

mu_r_sure_bl = mu_r_equil + sig2_hat_r @ v.T @ \
             np.linalg.solve(v @ sig2_hat_r @ v.T, i - v @ mu_r_equil)
sig2_r_sure_bl = (1 + 1 / tau) * sig2_hat_r - (1 / tau) * sig2_hat_r @ v.T\
               @ np.linalg.solve(v @ sig2_hat_r @ v.T, v @ sig2_hat_r)

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step09): Compare posterior parameters from point views

# +
k_ = len(v)  # view variables dimension
v_point = v
z_point = i

mu_r_point, sig2_r_point = min_rel_entropy_normal(mu_r_equil, sig2_hat_r,
                                                  v_point, z_point, v_point,
                                                  np.zeros((k_)))
# -

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_bl_equilibrium_ret-implementation-step10): Compute posterior parameters from distributional views (Minimum Relative Entropy)

# +
v_mre = v
v_sig_mre = np.eye(n_)
imre = i
sig2_i_mumre = sig2_hat_r

mu_r_mre, sig2_r_mre = min_rel_entropy_normal(mu_r_equil, sig2_hat_r, v_mre,
                                              imre, v_sig_mre, sig2_i_mumre)

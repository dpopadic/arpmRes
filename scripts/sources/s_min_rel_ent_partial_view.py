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

# # s_min_rel_ent_partial_view [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_min_rel_ent_partial_view&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExViewTheoryPart).

# +
import numpy as np

from scipy.stats import chi2

from arpym.estimation import cov_2_corr
from arpym.tools import mahalanobis_dist
from arpym.views import min_rel_entropy_normal, rel_entropy_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_partial_view-parameters)

mu_x_base = np.array([0.26, 0.29, 0.33])  # base expectation
sig2_x_base = np.array([[0.18, 0.11, 0.13],
                        [0.11, 0.23, 0.16],
                        [0.13, 0.16, 0.23]])  # base covariance
v_mu = np.array([[1, -1, 0], [0, 1, -1]])  # view on expectation matrix
v_sig = np.array([[-3, -1, -1], [-1, 2, 1]])  # view on covariance matrix
# view quantification parameters
mu_view = np.array([1.02, -0.50])
sig2_view = np.array([[0.19, 0.09], [0.09, 0.44]])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_partial_view-implementation-step01): Compute effective ranks corresponding to the pick matrices

# +

def eff_rank(s2):
    lam2_n, _ = np.linalg.eig(s2)
    w_n = lam2_n / np.sum(lam2_n)
    return np.exp(- w_n @ np.log(w_n))


eff_rank_v_mu = eff_rank(cov_2_corr(v_mu @ sig2_x_base @ v_mu.T)[0])
eff_rank_v_sig = eff_rank(cov_2_corr(v_sig @ sig2_x_base @ v_sig.T)[0])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_partial_view-implementation-step02): Compute updated parameters

mu_x_upd, sig2_x_upd = min_rel_entropy_normal(mu_x_base, sig2_x_base, v_mu,
                                              mu_view, v_sig, sig2_view)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_partial_view-implementation-step03): Compute projectors

# +
k_ = len(mu_view)  # view variables dimension
n_ = len(mu_x_base)  # market dimension

v_mu_inv = sig2_x_base @ v_mu.T @ np.linalg.solve(v_mu @ sig2_x_base @ v_mu.T,
                                                  np.identity(k_))
v_sig_inv = sig2_x_base @ v_sig.T @\
    (np.linalg.solve(v_sig @ sig2_x_base @ v_sig.T, np.identity(k_)))
p_mu = np.eye(n_) - v_mu_inv @ v_mu
p_mu_c = v_mu_inv @ v_mu
p_sig = np.eye(n_) - v_sig_inv @ v_sig
p_sig_c = v_sig_inv @ v_sig
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_partial_view-implementation-step04): Compute Mahalanobis distance and p-value

mah_distance = mahalanobis_dist(mu_x_upd, mu_x_base, sig2_x_base)
p_value = 1 - chi2.cdf(mah_distance, n_)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_partial_view-implementation-step05): Compute relative entropy and sensitivity to the views

rel_entropy = rel_entropy_normal(mu_x_upd, sig2_x_upd, mu_x_base, sig2_x_base)
sens = np.linalg.solve(v_mu @ sig2_x_base @ v_mu.T, v_mu @
                       (mu_x_upd - mu_x_base))

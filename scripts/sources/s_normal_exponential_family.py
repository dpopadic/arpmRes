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

# # s_normal_exponential_family [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_normal_exponential_family&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-normal-exponential-family-dist).

# +
import numpy as np

from arpym.statistics import normal_canonical
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_normal_exponential_family-parameters)

mu = np.array([1, 2])
sig2 = np.array([[1, 2], [2, 9]])
mu_xz = np.array([1, 2, 0.5])
sig2_xz = np.array([[1, 2, 1], [2, 9, 2], [1, 2, 2]])
z = np.array([1])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_normal_exponential_family-implementation-step01): Compute natural parameters and log-partition for X

theta_mu, theta_sig = normal_canonical(mu, sig2)
psi = -1/4 * theta_mu.T@np.linalg.solve(sig2, theta_mu) - \
      1/2*np.log(np.linalg.det(2*theta_sig))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_normal_exponential_family-implementation-step02): Compute natural parameters for XZ

theta_mu_xz, theta_sig_xz = normal_canonical(mu_xz, sig2_xz)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_normal_exponential_family-implementation-step03): Compute natural parameters for X|z using natural parameters

n_ = sig2.shape[0]
theta_mu_x_z = theta_mu_xz[:n_] + 2 * theta_sig_xz[n_:, :n_].T @ z
theta_sig_x_z = theta_sig_xz[:n_, :n_]

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_normal_exponential_family-implementation-step04): Compute natural parameters for X|z using normal parameters

# +
mu_z = mu_xz[n_:]
sig2_z = sig2_xz[n_:, n_:]
sig_xz = sig2_xz[n_:, :n_].T

# conditional distribution parameters
mu_x_z = mu + sig_xz @ np.linalg.solve(sig2_z, z - mu_z)
sig2_x_z = sig2 - sig_xz @ np.linalg.solve(sig2_z, sig_xz.T)

# conditional distribution natural parameters
theta_mu_x_z_1, theta_sig_x_z_1 = normal_canonical(mu_x_z, sig2_x_z)

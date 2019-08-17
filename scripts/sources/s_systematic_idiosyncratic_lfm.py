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

# # s_systematic_idiosyncratic_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_systematic_idiosyncratic_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmsys-id).

# +
import numpy as np

from arpym.estimation import factor_analysis_paf
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_systematic_idiosyncratic_lfm-parameters)

sigma2_u = np.array([[0.4, 0.],
                     [0., 0.4]])  # residual covariance
sigma2_h = 1  # factor covariance
beta = np.array([-np.sqrt(0.6), -np.sqrt(0.6)])  # loadings

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_systematic_idiosyncratic_lfm-implementation-step01): Compute the covariance of X and H

# cross-covariance of target and factor
sigma2_xh = beta * sigma2_h

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_systematic_idiosyncratic_lfm-implementation-step02): Compute the covariance of X

# target covariance
if len(beta.shape) == 1:
    k_ = 1
else:
    k_ = beta.shape[1]  # number of factors
if k_ == 1:
    sigma2_x = beta.reshape(-1, 1) * sigma2_h @ beta.reshape(1, -1) + sigma2_u
else:
    sigma2_x = beta @ sigma2_h @ beta.T + sigma2_u

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_systematic_idiosyncratic_lfm-implementation-step03): Compute factor loadings

# PAF loadings and residual std. deviations
beta_, delta2 = factor_analysis_paf(sigma2_x, k_=k_, eps=1e-5)
if k_ == 1:
    beta_ = beta_.reshape(-1, 1)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_systematic_idiosyncratic_lfm-implementation-step04): Compute the inverse of the covariance of X

omega2 = np.diagflat(1./delta2)
omega2_beta = omega2 @ beta_
# if k_ == 1:
#     omega2_beta = omega2_beta.reshape(-1, 1)
sigma2_x_inv = omega2 - omega2_beta @ \
    np.linalg.solve(beta_.T @ omega2_beta + np.eye(k_), omega2_beta.T)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_systematic_idiosyncratic_lfm-implementation-step05): Compute the r-squared

r2 = np.trace(beta_.T @ sigma2_x_inv @ beta_)/k_

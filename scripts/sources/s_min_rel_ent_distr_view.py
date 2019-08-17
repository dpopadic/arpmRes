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

# # s_min_rel_ent_distr_view [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_min_rel_ent_distr_view&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExViewTheoryDistr).

# +
import numpy as np

from arpym.views import min_rel_entropy_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_distr_view-parameters)

# +
mu_x_base = np.array([0.26, 0.29, 0.33])  # base case expectation
sig2_x_base = np.array([[0.18, 0.11, 0.13],
                        [0.11, 0.23, 0.16],
                        [0.13, 0.16, 0.23]])  # base case covariance

v = np.array([[1, -1, 0], [0, 1, -1]])  # view matrix

mu_z_view = np.array([1.02, -0.50])  # expectation of the view variables

sig2_z_view = np.array([[0.35, -0.40],
                        [-0.40, 0.21]])  # covariance of the view variables
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_distr_view-implementation-step01): Compute base parameters of the view variables

mu_z_base = v @ mu_x_base
sig2_z_base = v @ sig2_x_base @ v.T

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_distr_view-implementation-step02): Compute distributional view updated parameters

mu_x_upd, sig2_x_upd = min_rel_entropy_normal(mu_x_base, sig2_x_base, v,
                                              mu_z_view, v, sig2_z_view)

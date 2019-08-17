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

# # s_min_rel_ent_point_view [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_min_rel_ent_point_view&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExViewTheoryPoint).

# +
import numpy as np

from arpym.views import min_rel_entropy_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_point_view-parameters)

mu_base = np.array([0.26, 0.29, 0.33])  # base expectation
sig2_base = np.array([[0.18, 0.11, 0.13],
                     [0.11, 0.23, 0.16],
                     [0.13, 0.16, 0.23]])  # base covariance
v = np.array([[1, -1, 0], [0, 1, -1]])  # view matrix
z_view = np.array([1.02, -0.5])  # point view

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_min_rel_ent_point_view-implementation-step01): Compute point view updated parameters

# +
k_, n_ = v.shape  # market and view dimension

mu_upd, sig2_upd = min_rel_entropy_normal(mu_base, sig2_base, v, z_view, v,
                                          np.zeros((k_)))

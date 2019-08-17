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

# # s_views_sorted_exp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_views_sorted_exp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-example-fpviews-ranking).

# +
import numpy as np

from arpym.views import min_rel_entropy_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_views_sorted_exp-parameters)

# scenarios of market variables
x = np.array([[0.2, 1.7, 2, 3.4], [5, 3.4, -1.3, 1]]).T
mu_view = np.array([0])
p_base = np.ones(x.shape[0]) / x.shape[0]  # base flexible probabilities
v = np.array([1, - 1])
c = 0.2  # confidence level

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_views_sorted_exp-implementation-step01): Compute parameters specifying the inequality constraints

# +
z_ineq = (x @ v).reshape(1, x.shape[0])
mu_view_ineq = mu_view

exp_v = p_base @ x[:, ]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_views_sorted_exp-implementation-step02): Compute updated probabilities

p_upd = min_rel_entropy_sp(p_base, z_ineq, mu_view_ineq, None, None,
                           normalize=False)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_views_sorted_exp-implementation-step03): Compute additive/multiplicative confidence-weighted probabilities

p_c_add = c * p_upd + (1 - c) * p_base
p_c_mul = p_upd ** c * p_base ** (1 - c) /\
    np.sum(p_upd ** c * p_base ** (1 - c))

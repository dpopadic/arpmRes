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

# # s_views_correlations [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_views_correlations&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-example-fpviews-correlation).

# +
import numpy as np

from arpym.statistics import meancov_sp
from arpym.views import min_rel_entropy_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_views_correlations-parameters)

# +
# scenarios of market variables
x = np.array([[0.2, 1.7, 2, 3.4], [5, 3.4, -1.3, 1]]).T
p_base = np.ones(x.shape[0]) / x.shape[0]  # base flexible probabilities
rho_view = 0.2  # correlation
c = 0.2  # confidence level


def v_1(y):
    return np.array(y[:, 0] * np.exp([y[:, 1]]))  # view function


def v_2(y):
    return np.array(2 * y[:, 0] - np.exp([y[:, 1]]))  # view function


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_views_correlations-implementation-step01): Compute parameters specifying the constraints

mu_base_1, sig2_base_1 = meancov_sp(v_1(x).T, p_base)
sig_base_1 = np.sqrt(sig2_base_1)

mu_base_2, sig2_base_2 = meancov_sp(v_2(x).T, p_base)
sig_base_2 = np.sqrt(sig2_base_2)

z_ineq = v_1(x) * v_2(x)
mu_view_ineq = (rho_view * sig_base_1 * sig_base_2 +
                mu_base_1 * mu_base_2).reshape(1, )

z_eq = np.vstack((v_1(x), v_2(x), v_1(x) ** 2, v_2(x) ** 2))
mu_view_eq = np.vstack((mu_base_1, mu_base_2, mu_base_1 ** 2 + sig2_base_1,
                        mu_base_2 ** 2 + sig2_base_2)).reshape(4, )
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_views_correlations-implementation-step02): Compute updated probabilities

p_upd = min_rel_entropy_sp(p_base, z_ineq, mu_view_ineq, z_eq, mu_view_eq,
                           normalize=False)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_views_correlations-implementation-step03): Compute additive/multiplicative confidence-weighted probabilities

p_c_add = c * p_upd + (1 - c) * p_base
p_c_mul = p_upd ** c * p_base ** (1 - c) /\
    np.sum(p_upd ** c * p_base ** (1 - c))

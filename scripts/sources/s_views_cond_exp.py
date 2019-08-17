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

# # s_views_cond_exp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_views_cond_exp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-example-fpviews-cond-exp).

# +
import numpy as np

from arpym.views import min_rel_entropy_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_views_cond_exp-parameters)

# +
# scenarios of market variables
x = np.array([[0.2, 1.7, 2, 3.4], [5, 3.4, -1.3, 1]]).T
p_base = np.ones(x.shape[0]) / x.shape[0]  # base flexible probabilities
mu_view = 0  # view on expectation
c_view = 0.7  # view on CVaR
c = 0.3  # confidence level


def v(y):
    return np.array(2 * y[:, 0] - y[:, 1])  # view function


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_views_cond_exp-implementation-step01): Compute parameters specifying the constraints

j_ = x.shape[0]
index = np.array([i for i in range(j_)])
z = v(x)
v_x = np.sort(v(x))


def indicator(y, a):
    return np.array([1 if y_j <= a else 0 for y_j in y])


z_eq = np.zeros((j_, 2, j_))

for i in range(j_):
    z_eq[i] = np.vstack((v_x * indicator(index, i), indicator(index, i)))

mu_view_eq_c = np.vstack((c_view * mu_view, c_view)).reshape(2, )
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_views_cond_exp-implementation-step02): Compute covariance matrices and effective ranks

# +

def eff_rank(s2):
    lam2_n, _ = np.linalg.eig(s2)
    w_n = lam2_n / np.sum(lam2_n)
    return np.exp(- w_n @ np.log(w_n))


covariance = np.zeros((j_, 2, 2))
effrank = np.zeros(j_)

for i in range(j_):
    z_i = z_eq[i]
    covariance[i] = np.cov(z_i)
    if np.linalg.matrix_rank(covariance[i]) > 1:
        effrank[i] = eff_rank(np.corrcoef(z_i))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_views_cond_exp-implementation-step03): Compute updated probabilities

# +
i_san_check = np.where(effrank > 1)[0]

p_upd_i = np.zeros((j_, j_))
entropy = np.zeros(j_)

for i in range(j_):
    if i in i_san_check:
        p_upd_i[i] = min_rel_entropy_sp(p_base, None, None, z_eq[i],
                                        mu_view_eq_c, normalize=False)
        entropy[i] = p_upd_i[i] @ np.log(p_upd_i[i] / p_base)

p_upd_san = p_upd_i[i_san_check]
p_upd_ihat = p_upd_san[np.argmin(entropy[i_san_check])]
p_upd = p_upd_ihat[np.argsort(np.argsort(v(x)))]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_views_cond_exp-implementation-step04): Compute additive/multiplicative confidence-weighted probabilities

p_c_add = c * p_upd + (1 - c) * p_base
p_c_mul = p_upd ** c * p_base ** (1 - c) /\
    np.sum(p_upd ** c * p_base ** (1 - c))

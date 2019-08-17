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

# # s_factors_selection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_factors_selection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmselect).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics import objective_r2
from arpym.tools import naive_selection, forward_selection, \
                        backward_selection, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-parameters)

n_ = 1  # target variable dimension
m_ = 50  # number of factors

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step01): Generate random positive definite matrix

sigma = np.random.randn(n_ + m_, n_ + m_ + 1)
sigma2_xz = sigma @ sigma.T / (n_ + m_ + 1)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step02): Setup objective function

def g(s_k):
    return objective_r2(s_k, sigma2_xz, n_)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step03): Select the best factors via the naive, forward and backward stepwise routines

s_star_naive = naive_selection(g, m_)
s_star_fwd = forward_selection(g, m_)
s_star_bwd = backward_selection(g, m_)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step04): Compute the r2 based on number of factors

g_s_star_naive = np.zeros(m_)
g_s_star_fwd = np.zeros(m_)
g_s_star_bwd = np.zeros(m_)
for k in range(m_):
    g_s_star_naive[k] = g(s_star_naive[k])
    g_s_star_fwd[k] = g(s_star_fwd[k])
    g_s_star_bwd[k] = g(s_star_bwd[k])

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()
plt.plot(g_s_star_naive)
plt.plot(g_s_star_fwd, color='red')
plt.plot(g_s_star_bwd, color='blue')
plt.legend(['naive', 'forward stepwise', 'backward stepwise'])
plt.xlabel('number of risk factors Z')
plt.ylabel('R-square')
plt.title('n-choose-k routines comparison')
add_logo(fig)

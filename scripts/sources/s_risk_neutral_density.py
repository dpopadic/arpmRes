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

# # s_risk_neutral_density [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_risk_neutral_density&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comprnnumsdf).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.stats import norm, lognorm

from arpym.pricing import bsm_function
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-parameters)

# +
mu_tnow = 1e-3  # location parameter of lognormal distribution
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step00): Upload data

# +
path = '../../../databases/temporary-databases/'
db_simcall = pd.read_csv(path+'db_simcall.csv', index_col=0)
s_thor = db_simcall.s_thor.values
v_ad_tnow = db_simcall.v_ad_tnow.values
db_tools = pd.read_csv(path+'db_simcall_tools.csv', index_col=0)
s_tnow = db_tools.s_tnow.values[0]
tau_hor = db_tools.tau_hor.values[0]
gamma = db_tools.gamma.values[0]
r = db_tools.r.values[0]
sigma_tnow = db_tools.sigma_tnow.values[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step01): Compute the normalized underlying probabilities

# +
scale_p = s_tnow * np.exp((mu_tnow - sigma_tnow ** 2 / 2) * tau_hor)
p = lognorm.cdf(s_thor + gamma / 2, sigma_tnow*np.sqrt(tau_hor), scale=scale_p) - \
    lognorm.cdf(s_thor - gamma / 2, sigma_tnow*np.sqrt(tau_hor), scale=scale_p)
p = p / np.sum(p)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step02): Compute risk-neutral probabilities

# +
q = v_ad_tnow*np.exp(-tau_hor*r)
q = q / np.sum(q)
q = q.reshape(-1)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step03): Compute pdf

# +
s_low = s_thor[0]
s_up = s_thor[-1]  # upper bound of underlying at the horizon
s_thor_ = np.linspace(s_low, s_up, 100000)
p_ = lognorm.pdf(s_thor_, sigma_tnow * np.sqrt(tau_hor), scale=scale_p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step04): Compute risk-neutral pdf

# +
scale_q = s_tnow * np.exp((r - sigma_tnow ** 2 / 2) * tau_hor)
q_ = lognorm.pdf(s_thor_, sigma_tnow * np.sqrt(tau_hor), scale=scale_q)
# -

# ## Plots

# +
fig = plt.figure()
plt.style.use('arpm')

# plot histograms
plt.bar(s_thor, p / gamma, width=gamma, facecolor='none', edgecolor='b',
        label='Simulated market probability')
plt.bar(s_thor, q / gamma, width=gamma, facecolor='none', edgecolor='g',
        linestyle='--', label='Simulated risk neutral probability')

# plot pdfs
plt.plot(s_thor_, p_, 'b', lw=1.5, label='Analytical market density')
plt.plot(s_thor_, q_, 'g--', lw=1.5, label='Analytical risk neutral density')
plt.xlabel('$S_{t_end{hor}}$')
plt.ylabel('Density function')
plt.legend()

add_logo(fig)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_dependence_structure_call_put [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_dependence_structure_call_put&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-smooth-approx-call-put).

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from arpym.statistics import cop_marg_sep, meancov_sp, simulate_normal, schweizer_wolff
from arpym.estimation import cov_2_corr
from arpym.tools.regularized_payoff import regularized_payoff
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-parameters)

mu = [1, 3]  # location parameter
# dispersion parameters
rho_12 = -0.2
sig_1 = 0.5
sig_2 = 0.3
j_ = 10**2  # number of simulations
k_strk = 2.71  # strike
h = 0.5  # aproximation level

# # Step 1: Generate log-normal samples

sig2 = np.array([[sig_1**2, rho_12*sig_1*sig_2],
                 [rho_12*sig_1*sig_2, sig_2**2]])
# jointly lognormal samples
x = np.exp(simulate_normal(mu, sig2, j_))

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-implementation-step01): Compute call and put payoffs

v_put = np.maximum(k_strk - x[:, 0], 0)
v_call = np.maximum(x[:, 0] - k_strk, 0)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-implementation-step02):  Compute regularized call and put payoffs

v_put_h = regularized_payoff(x[:, 0], k_strk, h, method='put')  # regularized payoff of put option
v_call_h = regularized_payoff(x[:, 0], k_strk, h, method='call')  # regularized payoff of call option

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-implementation-step03):  Compute Schweizer and Wolff

sw_x = schweizer_wolff(x)
sw_x1v_put_h = 1
sw_x1v_call_h = 1
sw_x2v_put_h = schweizer_wolff(x)
sw_x2v_call_h = schweizer_wolff(x)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-implementation-step04):  Compute Kendal's tau

kend_x = 2 / np.pi * np.arcsin(rho_12)
kend_x1v_put_h = -1
kend_x1v_call_h = 1
kend_x2v_put_h = -2 / np.pi * np.arcsin(rho_12)
kend_x2v_call_h = 2 / np.pi * np.arcsin(rho_12)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-implementation-step05): Compute Spearman rho

# compute grades scenarios
u_x, _, _ = cop_marg_sep(x)
u_v_p, _, _ = cop_marg_sep(v_put_h)
u_v_c, _, _ = cop_marg_sep(v_call_h)

# Spearman rho
_, cov_u = meancov_sp(np.c_[u_x[:, 0], u_v_p, u_v_c, u_x[:, 1]])
spear, _ = cov_2_corr(cov_u)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_dependence_structure_call_put-implementation-step06): Compute correlation

cov = np.cov(np.c_[x[:, 0], v_put_h, v_call_h, x[:, 1]].T)
corr, _ = cov_2_corr(cov)

# ## Plots

# +
plt.style.use('arpm')
violet = [170/255,	 85/255, 187/255]
teal = [60/255,	 149/255, 145/255]

f = plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
spec = gridspec.GridSpec(ncols=2, nrows=2)

f_ax1 = f.add_subplot(spec[:, 0])
plt.plot(np.sort(x[:, 0]), -np.sort(-v_put), color=violet, lw=2, label=r'$V_1^{put}$')
plt.plot(np.sort(x[:, 0]), -np.sort(-v_put_h), color=teal, lw=2, label=r'$V_{1;h}^{put}$')
plt.legend(loc=9)
plt.xlabel('Strike')
plt.ylabel('Payoff')
plt.title('Put option')

f_ax2 = f.add_subplot(spec[:, 1])
plt.plot(np.sort(x[:, 0]), np.sort(v_call), color=violet, lw=2, label=r'$V_1^{call}$')
plt.plot(np.sort(x[:, 0]), np.sort(v_call_h), color=teal, lw=2, label=r'$V_{1;h}^{call}$')
plt.legend(loc=9)
plt.xlabel('Strike')
plt.ylabel('Payoff')
plt.title('Call option')

add_logo(f, location=4, set_fig_size=False)

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

# # s_location_estimators [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_location_estimators&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExermuEstimDist).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_location_estimators-parameters)

t_ = 15  # length of the time series
j_ = 10 ** 3  # number of simulations
mu = 2  # true value of the parameter

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_location_estimators-implementation-step01): Generate simulations of the time series of invariants

i_mu = stats.norm.rvs(mu, 1, (j_, t_))  # simulations

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_location_estimators-implementation-step02): Sample mean computations

# +
m_hat = np.mean(i_mu, axis=1)  # simulations
exp_m = np.mean(m_hat)  # expectation
bias2_m = (exp_m - mu) ** 2  # square bias
inef_m = np.std(m_hat, ddof=1)  # inefficiency

l_m = (m_hat - mu) ** 2  # loss
er_m = np.mean(l_m)  # error
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_location_estimators-implementation-step03): Product estimator computations

# +
pi_hat = i_mu[:, 0] * i_mu[:, -1]  # simulations
exp_pi = np.mean(pi_hat)  # expectation
bias2_pi = (exp_pi - mu) ** 2  # square bias
inef_pi = np.std(pi_hat, ddof=1)  # inefficiency

l_pi = (pi_hat - mu) ** 2  # loss
er_pi = np.mean(l_pi)  # error
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_location_estimators-implementation-step04): Constant estimator

# +
k_hat = 3*np.ones(j_)  # simulations
exp_k = np.mean(k_hat)  # expectation
bias2_k = (exp_k - mu) ** 2  # square bias
inef_k = np.std(k_hat, ddof=1)  # inefficiency

l_k = (k_hat - mu) ** 2  # loss
er_k = np.mean(l_k)  # error
# -

# ## Plots

# +
plt.style.use('arpm')

l_ = 125
x = np.linspace(mu - 4, mu+4, l_)

f_epsi = stats.norm.pdf(x, mu, 1)  # invariants' pdf

# histograms computations

# compute histogram
# sample mean histograms
m_hist, m_x = histogram_sp(m_hat)
# product estimator histograms
pi_hist, pi_x = histogram_sp(pi_hat)
# constant estimator histograms
k_hist, k_x = histogram_sp(k_hat.T, xi=np.arange(-33, 36))

l_m_hist, l_m_x = histogram_sp(l_m)
l_pi_hist, l_pi_x = histogram_sp(l_pi)
l_k_hist, l_k_x = histogram_sp(l_k, xi=np.arange(-33, 36))

colhist = [.8, .8, .8]
orange = [1, 0.4, 0]
green = [0.1, 0.8, 0]
dark = [0.2, 0.2, 0.2]
blue = [0, 0.4, 1]

# histogram of invariants
fig1 = plt.figure()
heps = plt.plot(x, f_epsi, color=blue, lw=1.5)
plt.plot([mu, mu], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlabel('$\epsilon$')
plt.title('TRUE (UNKNOWN) DISTRIBUTION')
epsT = '$\mu$ =  % 3.2f' % mu
plt.text(np.min(x)+0.1, np.max(f_epsi)*0.95 - 0.001,
         '$\epsilon_{t} \sim$ N(%s,$\sigma^2$ = 1)' % epsT, color='k',
         horizontalalignment='left')
add_logo(fig1, location=1)
plt.tight_layout()

# histograms of estimators
fig2, ax = plt.subplots(1, 3)
# sample mean
plt.sca(ax[0])
hm = plt.bar(m_x, m_hist, width=m_x[1]-m_x[0], facecolor=colhist,
             edgecolor='k')
plt.plot([exp_m, exp_m], [0, 0], color=orange, marker='o', markersize=6,
         markerfacecolor=orange)
plt.plot([mu, exp_m], [0, 0], color=orange, lw=6)
plt.plot([exp_m - inef_m, exp_m + inef_m], [np.max(m_hist)*0.02,
         np.max(m_hist)*0.02], color=blue, lw=4)
plt.plot([mu, mu], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlim([np.percentile(m_hat, 100 * 0.0001), np.percentile(m_hat,
          100 * 0.9999)])
plt.xlabel('sample mean')

# constant estimator
plt.sca(ax[1])
hk = plt.bar(k_x, k_hist / np.sum(k_hist), width=0.3,
             facecolor=colhist, edgecolor='k')
plt.plot([exp_k, exp_k], [0, 0], color=orange, marker='o', markersize=6,
         markerfacecolor=orange)
bias_plot = plt.plot([mu, exp_k], [0, 0], color=orange, lw=4)
inef_plot = plt.plot([exp_k - inef_k, exp_k + inef_k], [np.max(k_hist)*0.02,
                     np.max(k_hist)*0.02], color=blue, lw=4)
plt.plot([mu, mu], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlim([min([mu, 3]) - abs((mu - 3))*1.1, max([mu, 3]) + abs((mu - 3))*1.1])
plt.xlabel('constant')
plt.title('ESTIMATORS DISTRIBUTION')

# product estimator
plt.sca(ax[2])
hpi = plt.bar(pi_x, pi_hist, width=pi_x[1]-pi_x[0], facecolor=colhist,
              edgecolor='k')
plt.plot([exp_pi, exp_pi], [0, 0], color=orange, marker='o', markersize=6,
         markerfacecolor=orange)
plt.plot([mu, exp_pi], [0, 0], color=orange, lw=4)
plt.plot([exp_pi - inef_pi, exp_pi + inef_pi], [np.max(pi_hist)*0.02,
         np.max(pi_hist)*0.02], color=blue, lw=4)
plt.plot([mu, mu], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlim([np.percentile(pi_hat, 100 * 0.001), np.percentile(pi_hat,
          100 * 0.999)])
plt.xlabel('first-last product')
plt.legend(handles=[bias_plot[0], inef_plot[0]], labels=['bias', 'ineff.'])
add_logo(fig2, location=5, size_frac_x=1/5)
plt.tight_layout()

# histograms of square losses
fig3, ax = plt.subplots(1, 3)
# sample mean
plt.sca(ax[0])
hLm = plt.bar(l_m_x, l_m_hist, width=l_m_x[1]-l_m_x[0],
              facecolor=colhist, edgecolor='k')
plt.plot([0, bias2_m], [0.002, 0.002], color=orange, lw=5)
plt.plot([bias2_m, er_m], [0.002, 0.002], color=blue, lw=5)
plt.plot([0, er_m], [np.max(l_m_hist)*0.0275, np.max(l_m_hist)*0.0275],
         color=dark, lw=5)
plt.plot([0, 0], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlim([-max(l_m)*0.005, np.percentile(l_m, 100 * 0.95)])
plt.xlabel('sample mean')
# constant estimator
plt.sca(ax[1])
hLk = plt.bar(l_k_x, l_k_hist / np.sum(l_k_hist), width=0.1,
              facecolor=colhist, edgecolor='none')
plt.plot([0, bias2_k], [0.001, 0.001], color=orange, lw=5)
plt.plot([bias2_k, er_k], [0.001, 0.001], color=blue, lw=5)
plt.plot([0, er_k], [np.max(l_k_hist)*0.0275, np.max(l_k_hist)*0.0275],
         color=dark, lw=5)
plt.plot([0, 0], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlim([-0.01, 1.25*(mu - 3) ** 2])
plt.xlabel('constant')
plt.title('LOSS DISTRIBUTION')
# product estimator
plt.sca(ax[2])
hLpi = plt.bar(l_pi_x, l_pi_hist, width=l_pi_x[1]-l_pi_x[0],
               facecolor=colhist, edgecolor='k')
bias_plot = plt.plot([0, bias2_pi], [0.001, 0.001], color=orange, lw=5,
                     zorder=2)
inef_plot = plt.plot([bias2_pi, er_pi], [0.001, 0.001], color=blue, lw=5,
                     zorder=1)
error_plot = plt.plot([0, er_pi], [np.max(l_pi_hist)*0.0275,
                 np.max(l_pi_hist)*0.0275], color=dark, lw=5, zorder=1)
plt.plot([0, 0], [0, 0], color=green, marker='o', markersize=6,
         markerfacecolor=green)
plt.xlim([-max(l_pi)*0.005, np.percentile(l_pi, 100 * 0.95)])
plt.xlabel('first-last product')
plt.legend(handles=[error_plot[0], bias_plot[0], inef_plot[0]],
           labels=['error', 'bias$^2$', 'ineff.$^2$'])
add_logo(fig3, location=5, size_frac_x=1/5)
plt.tight_layout()

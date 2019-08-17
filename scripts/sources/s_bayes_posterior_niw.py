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

# # s_bayes_posterior_niw [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bayes_posterior_niw&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFigNIWposterior).

# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wishart, invwishart, norm, t

from arpym.statistics import simulate_niw
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-parameters)

mu_pri = 0.1  # prior expectation
sigma2_pri = 2.  # prior dispersion
t_pri = 7.  # confidence on mu_pri
v_pri = 5.  # confidence on sigma2_pri
mu = 1.5  # true value of mu
sigma2 = 4.  # true value of sigma
t_ = 6  # length of the time series
j_ = 5000  # number of simulations
k_ = 500  # number of grid points

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step01): Generate the time series and compute the sample mean and covariance estimators

epsi = norm.rvs(mu, np.sqrt(sigma2), t_)
mu_hat = np.mean(epsi)
sigma2_hat = np.var(epsi)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step02): Compute the parameters of the posterior distribution

mu_pos = (t_pri / (t_pri + t_)) * mu_pri + (t_ / (t_pri + t_)) * mu_hat
sigma2_pos = (v_pri / (v_pri + t_)) * sigma2_pri + \
             (t_ / (v_pri + t_)) * sigma2_hat + \
             (mu_pri - mu_hat) ** 2 / ((v_pri + t_) * (1 / t_ + 1 / t_pri))
t_pos = t_pri + t_
v_pos = v_pri + t_

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step03): Compute the mean and standard deviations of the sample, prior and posterior distributions of Sigma2

exp_sigma2_hat = wishart.mean(t_ - 1, sigma2_hat / t_)
std_sigma2_hat = np.sqrt(wishart.var(t_ - 1, sigma2_hat / t_))
exp_sigma2_pri = invwishart.mean(v_pri, v_pri * sigma2_pri)
std_sigma2_pri = np.sqrt(invwishart.var(v_pri, v_pri * sigma2_pri))
exp_sigma2_pos = invwishart.mean(v_pos, v_pos * sigma2_pos)
std_sigma2_pos = np.sqrt(invwishart.var(v_pos, v_pos * sigma2_pos))

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step04): Compute marginal pdfs of the sample, prior and posterior distributions of Sigma2

# +
s_max = np.max([exp_sigma2_hat + 3. * std_sigma2_hat,
                exp_sigma2_pri + 3. * std_sigma2_pri,
                exp_sigma2_pos + 3. * std_sigma2_pos])
s = np.linspace(0.01, s_max, k_)  # grid

f_sigma2_hat = wishart.pdf(s, t_ - 1, sigma2_hat / t_)  # sample pdf
f_sigma2_pri = invwishart.pdf(s, v_pri, v_pri * sigma2_pri)  # prior pdf
f_sigma2_pos = invwishart.pdf(s, v_pos, v_pos * sigma2_pos)  # posterior pdf
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step05): Compute the pdf of the sample, prior and posterior distributions of M

# +
m_min = np.min([mu_hat - 3. * np.sqrt(sigma2_hat / t_),
                mu_pri - 3. * np.sqrt(sigma2_pri / t_pri),
                mu_pos - 3. * np.sqrt(sigma2_pos / t_pos)])
m_max = np.max([mu_hat + 3. * np.sqrt(sigma2_hat / t_),
                mu_pri + 3. * np.sqrt(sigma2_pri / t_pri),
                mu_pos + 3. * np.sqrt(sigma2_pos / t_pos)])
m = np.linspace(m_min, m_max, k_)  # grid

f_m_hat = norm.pdf(m, mu_hat, np.sqrt(sigma2_hat / t_))  # sample pdf
f_m_pri = t.pdf((m - mu_pri) / np.sqrt(sigma2_pri / t_pri),
                v_pri) / np.sqrt(sigma2_pri / t_pri)  # prior pdf
f_m_pos = t.pdf((m - mu_pos) / np.sqrt(sigma2_pos / t_pos),
                v_pos) / np.sqrt(sigma2_pos / t_pos)  # posterior pdf
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step06): Compute the joint pdf of the sample, prior and posterior distributions of (M, Sigma2)

# +
f_hat = np.outer(f_sigma2_hat, f_m_hat)  # sample pdf

f_pri = np.zeros((k_, k_))
f_pos = np.zeros((k_, k_))
for k in range(k_):
    # prior pdf
    f_pri[k, :] = norm.pdf(m, mu_pri, np.sqrt(s[k] / t_pri)) * f_sigma2_pri[k]
    # posterior pdf
    f_pos[k, :] = norm.pdf(m, mu_pos, np.sqrt(s[k] / t_pos)) * f_sigma2_pos[k]
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_posterior_niw-implementation-step07): Generate random NIW variables

m_sim, s2_sim = simulate_niw(mu_pos, t_pos, sigma2_pos, v_pos, j_)

# ## Plots

# +
plt.style.use('arpm')

# colors settings
color_pri = [0.2, 0.3, 1]
color_pos = [0.9, 0.3, 0.1]
color_sam = [0.1, 0.7, 0.1]

# pdf of Sigma2
fig = plt.figure()
plt.plot(s, f_sigma2_hat, lw=1.5, color=color_sam)
plt.plot(s, f_sigma2_pri, lw=1.5, color=color_pri)
plt.plot(s, f_sigma2_pos, lw=1.5, color=color_pos)
plt.xlabel('$\Sigma^2$')
# dummy plots for generating legend
ax = plt.gca()
shx = ax.get_xlim()
shy = ax.get_ylim()
sh1 = ax.plot(shx[0], shy[0], color=color_sam,
              lw=1.5, marker='', label='sample')
sh2 = ax.plot(shx[0], shy[0], color=color_pri,
              lw=1.5, marker='', label='prior')
sh3 = ax.plot(shx[0], shy[0], color=color_pos,
              lw=1.5, marker='', label='posterior')
plt.legend()

add_logo(fig)
plt.tight_layout()

# pdf of M
fig = plt.figure()
plt.plot(m, f_m_hat, lw=1.5, color=color_sam)
plt.plot(m, f_m_pri, lw=1.5, color=color_pri)
plt.plot(m, f_m_pos, lw=1.5, color=color_pos)
plt.xlabel('$M$')
# dummy plots for generating legend
ax = plt.gca()
shx = ax.get_xlim()
shy = ax.get_ylim()
sh1 = ax.plot(shx[0], shy[0], color=color_sam,
              lw=1.5, marker='', label='sample')
sh2 = ax.plot(shx[0], shy[0], color=color_pri,
              lw=1.5, marker='', label='prior')
sh3 = ax.plot(shx[0], shy[0], color=color_pos,
              lw=1.5, marker='', label='posterior')
plt.legend()

add_logo(fig)
plt.tight_layout()

# contour plot of joint distribution
fig = plt.figure()
plt.contour(m, s, f_hat, 12, linewidths=1.5, colors=[color_sam])
plt.contour(m, s, f_pri, 12, linewidths=1.5, colors=[color_pri])
plt.contour(m, s, f_pos, 12, linewidths=1.5, colors=[color_pos])
plt.scatter(m_sim, s2_sim, 2, color=[color_pos])
plt.xlim([np.min(m), np.max(m)])
plt.ylim([np.min(s), np.max(s)])
plt.xlabel(r'$M$')
plt.ylabel(r'$\Sigma^2$')
plt.title('Joint pdf')
# dummy plots for generating legend
ax = plt.gca()
shx = ax.get_xlim()
shy = ax.get_ylim()
sh1 = ax.plot(shx[0], shy[0], color=color_sam,
              lw=1.5, marker='', label='sample')
sh2 = ax.plot(shx[0], shy[0], color=color_pri,
              lw=1.5, marker='', label='prior')
sh3 = ax.plot(shx[0], shy[0], color=color_pos,
              lw=1.5, marker='', label='posterior')
plt.legend()

add_logo(fig)
plt.tight_layout()

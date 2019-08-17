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

# # s_reg_lfm_bayes_posterior_niw [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_reg_lfm_bayes_posterior_niw&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExFactNIWposterior).

# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wishart, invwishart, norm, t

from arpym.statistics import simulate_niw
from arpym.tools import histogram_sp, add_logo
from arpym.estimation import fit_lfm_ols
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-parameters)

beta_pri = -1.5  # prior location parameter of the loadings
sigma2_pri = 6  # prior location parameter of the variance
sigma2_zpri = 2.5  # prior dispersion parameter of the loadings
t_pri = 6  # confidence on the prior loadings
v_pri = 6  # confidence on the prior variance
beta = 1.5  # true value of the loadings
sigma2 = 4  # real value of variance
t_ = 6  # length of the time series
k_ = 200  # number of grid points
j_ = 5000  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step01): Generate simulations of factor and conditional residual

z = norm.rvs(0, 1, t_)
u = norm.rvs(0, np.sqrt(sigma2), t_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step02): Compute simulations of conditional target variables

x = beta * z + u

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step03): Compute the least squares estimators

_, beta_hat, sigma2_hat, _ = fit_lfm_ols(x, z)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step04): Compute the parameters of the posterior distribution

sigma2_zhat = z @ z.T / t_
beta_pos = (beta_pri * t_pri * sigma2_zpri + beta_hat * t_ *
            sigma2_zhat) / (t_pri * sigma2_zpri + t_ * sigma2_zhat)
t_pos = t_pri + t_
v_pos = v_pri + t_
sigma2_zpos = (t_pri * sigma2_zpri + t_ * sigma2_zhat) / t_pos
sigma2_pos = (t_ * sigma2_hat + v_pri * sigma2_pri + t_pri * beta_pri *
              sigma2_zpri * beta_pri + t_ * beta_hat * sigma2_zhat *
              beta_hat.T - t_pos * beta_pos * sigma2_zpos * beta_pos.T) / v_pos

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step05): Compute the mean and standard deviations of the sample, prior and posterior distributions of Sigma2

exp_sigma2_hat = wishart.mean(t_ - 1, sigma2_hat / t_)
std_sigma2_hat = np.sqrt(wishart.var(t_ - 1, sigma2_hat / t_))
exp_sigma2_pri = invwishart.mean(v_pri, v_pri * sigma2_pri)
std_sigma2_pri = np.sqrt(invwishart.var(v_pri, v_pri * sigma2_pri))
exp_sigma2_pos = invwishart.mean(v_pos, v_pos * sigma2_pos)
std_sigma2_pos = np.sqrt(invwishart.var(v_pos, v_pos * sigma2_pos))

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step06): Compute marginal pdfs of the sample, prior and posterior distributions of Sigma2

# +
s_max = np.max([exp_sigma2_hat + 2. * std_sigma2_hat,
                exp_sigma2_pri + 2. * std_sigma2_pri,
                exp_sigma2_pos + 2. * std_sigma2_pos])
s = np.linspace(0.01, s_max, k_)

f_sigma2_hat = wishart.pdf(s, t_ - 1, sigma2_hat / t_)
f_sigma2_pri = invwishart.pdf(s, v_pri, v_pri * sigma2_pri)
f_sigma2_pos = invwishart.pdf(s, v_pos, v_pos * sigma2_pos)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step07): Compute the mean and standard deviations of the sample, prior and posterior distributions of B

exp_beta_hat = beta_hat
std_beta_hat = np.sqrt(sigma2_hat / (sigma2_zhat * t_))
exp_beta_pri = beta_pri
std_beta_pri = np.sqrt(sigma2_pri / (sigma2_zpri * t_pri) *
                       v_pri / (v_pri - 2.))
exp_beta_pos = beta_pos
std_beta_pos = np.sqrt(sigma2_pos / (sigma2_zpos * t_pos) *
                       v_pos / (v_pos - 2.))

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step08): Compute marginal pdfs of the sample, prior and posterior distributions of B

# +
b_min = np.min([exp_beta_hat - 4. * std_beta_hat,
                exp_beta_pri - 4. * std_beta_pri,
                exp_beta_pos - 4. * std_beta_pos])
b_max = np.max([exp_beta_hat + 4. * std_beta_hat,
                exp_beta_pri + 4. * std_beta_pri,
                exp_beta_pos + 4. * std_beta_pos])
b = np.linspace(b_min, b_max, k_)

f_beta_hat = norm.pdf(b, beta_hat, np.sqrt(sigma2_hat / (sigma2_zhat * t_)))
f_beta_pri = t.pdf((b - beta_pri) / np.sqrt(sigma2_pri /
                   (sigma2_zpri * t_pri)), v_pri) /\
                   np.sqrt(sigma2_pri / (sigma2_zpri * t_pri))
f_beta_pos = t.pdf((b - beta_pos) / np.sqrt(sigma2_pos /
                   (sigma2_zpos * t_pos)), v_pos) /\
                   np.sqrt(sigma2_pos / (sigma2_zpos * t_pos))
# -

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step09): Compute the joint pdf of the ols, prior and posterior distributions of (B,Sigma2)

# +
f_hat = np.outer(f_beta_hat, f_sigma2_hat)

f_pri = np.zeros((k_, k_))
f_pos = np.zeros((k_, k_))
for k in range(k_):
    f_pri[:, k] = norm.pdf(b, beta_pri, np.sqrt(s[k] /
                           (sigma2_zpri * t_pri))) * f_sigma2_pri[k]
    f_pos[:, k] = norm.pdf(b, beta_pos, np.sqrt(s[k] /
                           (sigma2_zpos * t_pos))) * f_sigma2_pos[k]
# -

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_posterior_niw-implementation-step10): Generate a sample from the posterior pdf

beta_sim, sigma2_sim = simulate_niw(beta_pos, sigma2_zpos * t_pos,
                                    sigma2_pos, v_pos, j_)

# ## Plot

# +
plt.style.use('arpm')

# colors settings
histcol = [.8, .8, .8]

# (colors for the pdf's)
color_pri = [0.2, 0.3, 1]
color_pos = [0.9, 0.3, 0.1]
color_sam = [0.1, 0.7, 0.1]

# simulations
fig1, ax = plt.subplots(2, 1)
plt.sca(ax[0])
NumBins = round(10 * np.log(j_))
# Mu
# compute histogram
n, xi = histogram_sp(beta_sim)
# plot empirical pdf (histogram) bars
bars = plt.bar(xi, n, width=xi[1]-xi[0], facecolor=histcol, edgecolor='k')
# superimpose analytical expectation
h = plt.plot(beta_pos, 0, '.', color='r', markersize=15)
plt.plot(b, f_beta_pos, 'r')  # superimpose analytical pdf
plt.title(r'posterior distribution')
plt.legend(['empirical pdf', 'analytical pdf'])

# Sigma2
plt.sca(ax[1])
n, xi = histogram_sp(sigma2_sim)
# plot empirical pdf (histogram)
bars = plt.bar(xi, n, width=xi[1]-xi[0], facecolor=histcol, edgecolor='k')
# superimpose analytical expectation
h = plt.plot(sigma2_pos, 0, '.', color='r', markersize=15)
plt.plot(s, f_sigma2_pos, 'r')  # superimpose analytical pdf
plt.title(r'$\Sigma^2$ posterior distribution')
add_logo(fig1)
plt.tight_layout()

# Sigma2
fig2 = plt.figure()
# pdf's
plt.plot(s, f_sigma2_hat, lw=1.5, color=color_sam)
plt.plot(s, f_sigma2_pri, lw=1.5, color=color_pri)
plt.plot(s, f_sigma2_pos, lw=1.7, color=color_pos)
# classical equivalents
plt.plot(sigma2_hat, 0, color=color_sam, marker='o', markersize=6,
         markerfacecolor=color_sam)
plt.plot(sigma2_pri, 0, color=color_pri, marker='o', markersize=6,
         markerfacecolor=color_pri)
plt.plot(sigma2_pos, 0, color=color_pos, marker='o', markersize=6,
         markerfacecolor=color_pos)
plt.xlabel(r'$\Sigma^2$')
plt.ylabel(r'$pdf\ \Sigma^2$')
plt.legend(['sample', 'prior', 'posterior'])
add_logo(fig2, location=5)
plt.tight_layout()

# B
fig3 = plt.figure()
# pdf's
plt.plot(b, f_beta_hat, lw=1.5, color=color_sam)
plt.plot(b, f_beta_pri, lw=1.5, color=color_pri)
plt.plot(b, f_beta_pos, lw=1.7, color=color_pos)
# classical equivalents
plt.plot(beta_hat, 0, color=color_sam, marker='o', markersize=6,
         markerfacecolor=color_sam)
plt.plot(beta_pri, 0, color=color_pri, marker='o', markersize=6,
         markerfacecolor=color_pri)
plt.plot(beta_pos, 0, color=color_pos, marker='o', markersize=6,
         markerfacecolor=color_pos)
plt.xlabel('B')
plt.ylabel(r'pdf B')
plt.legend(['sample', 'prior', 'posterior'])
add_logo(fig3, location=5)
plt.tight_layout()

# joint
fig4 = plt.figure()
plt.contour(b, s, f_hat.T, 12, colors=[color_sam])
plt.contour(b, s, f_pri.T, 12, colors=[color_pri])
plt.contour(b, s, f_pos.T, 12, colors=[color_pos])
plt.xlabel('B')
plt.ylabel(r'$\Sigma^2$')
plt.title('joint pdf')
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

add_logo(fig4)
plt.tight_layout()

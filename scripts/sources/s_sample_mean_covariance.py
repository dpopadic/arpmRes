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

# # s_sample_mean_covariance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_sample_mean_covariance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFigBayes1).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_sample_mean_covariance-parameters)

t_ = 10  # length of the time series
j_ = 1000  # number of simulations
mu = 1  # true value of the expectation
sigma2 = 4  # true value of the variance

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_sample_mean_covariance-implementation-step01): Generate simulations of invariant time series

i_theta = stats.norm.rvs(mu, np.sqrt(sigma2), size=[j_, t_])  # simulations

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_sample_mean_covariance-implementation-step02): Compute simulations of sample mean and sample variance estimators

m_hat = np.mean(i_theta, 1)  # sample mean
sigma2_hat = np.var(i_theta, axis=1, ddof=0)  # sample variance

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_sample_mean_covariance-implementation-step03): Compute pdf of sample mean estimator

# +
# analytical
x_m = np.arange(np.min(m_hat), np.max(m_hat)+0.01, 0.01)
f_m = stats.norm.pdf(x_m, mu, np.sqrt(sigma2 / t_))

# empirical histogram
m_hist, m_x = histogram_sp(m_hat)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_sample_mean_covariance-implementation-step04): Compute pdf of sample variance estimator

# +
# analytical
x_sigma2 = np.arange(min(sigma2_hat), max(sigma2_hat)+0.01, 0.01)
f_sigma2 = stats.wishart.pdf(x_sigma2, t_-1, sigma2 / t_)

# empirical histogram
sigma2_hist, sigma2_x = histogram_sp(sigma2_hat)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_sample_mean_covariance-implementation-step05): Compute the analytical joint pdf of sample mean and (co)variance

# +
f_joint = np.zeros((len(x_m), len(x_sigma2)))

for k1 in range(len(x_m)):
    # joint pdf
    f_joint[k1, :] = stats.norm.pdf(x_m[k1], mu, np.sqrt(sigma2 / t_))\
                     * stats.wishart.pdf(x_sigma2, t_-1, sigma2 / t_)
# -

# ## Plots

# +
plt.style.use('arpm')

blue = [0.25, 0.25, 1]
colhist = [.8, .8, .8]
grey = [0.6, 0.6, 0.6]

# pdf of invariants
x_epsi = np.arange(mu-3*np.sqrt(sigma2), mu + 3*np.sqrt(sigma2)+0.01, 0.01)
f_epsi = stats.norm.pdf(x_epsi, mu, np.sqrt(sigma2))

fig1 = plt.figure()

plt.plot(x_epsi, f_epsi, color=blue, lw=3)
plt.xlim([np.min(x_epsi), np.max(x_epsi)])
plt.ylim([0, 1.1*np.max(f_epsi)])
eps_string = '$\epsilon_t \sim N (\mu= %1.2f, \sigma^2= %1.2f )$' %\
        (mu, sigma2)
plt.text(np.max(x_epsi), np.max(f_epsi), eps_string, color='k',
         horizontalalignment='right')
plt.title('Invariants distribution (Normal)')
plt.xlabel('$\epsilon_{t}$')

add_logo(fig1, location=2)

# sample mean distribution
fig2 = plt.figure()

plt.bar(m_x, m_hist, width=m_x[1]-m_x[0], facecolor=colhist)
m_lim = [x_m[0], x_m[-1]]
ymax = max([np.max(m_hist), np.max(f_m)])
plt.xlim(m_lim)
plt.ylim([0, 1.1*ymax])
plt.plot(x_m, f_m, color=blue, lw=3)
plt.title('Sample mean distribution')
plt.legend(['empirical pdf', 'analytical pdf'])
plt.xlabel('$\hat{M}$')
plt.text(0.8*m_lim[1], 0.7*ymax,
         r'$\hat{M} \sim N (\mu,\frac{\sigma^{2}}{\overline{t}})$',
         horizontalalignment='right')
add_logo(fig2, location=2)

# sample covariance distribution
fig3 = plt.figure()
plt.bar(sigma2_x, sigma2_hist, width=sigma2_x[1]-sigma2_x[0],
        facecolor=colhist, edgecolor='k')
sigma2_lim = [x_sigma2[0], x_sigma2[-1]]
plt.xlim(sigma2_lim)
plt.ylim([0, 1.1*ymax])
plt.plot(x_sigma2, f_sigma2, color=blue, lw=3)
plt.title('Sample (co)variance distribution (Wishart distribution)')
plt.legend(['empirical pdf', 'analytical pdf'])
plt.xlabel('$\hat{\Sigma}^2$')
plt.text(0.8*sigma2_lim[1], 0.7*ymax,
         r'$\hat{\Sigma}^{2} \sim Wishart(\overline{t}-1,\frac{\sigma^{2}}{\overline{t}})$',
         horizontalalignment='right')
add_logo(fig3)

# joint distribution
fig4 = plt.figure()
esc = plt.plot(m_hat[:int(j_ / 2)], sigma2_hat[:int(j_ / 2)], markersize=4,
               color=grey, marker='.', linestyle='none')
xlimm = [np.percentile(m_hat, 100 * 0.001), np.percentile(m_hat, 100 * 0.999)]
ylimm = [np.min(sigma2_hat), np.percentile(sigma2_hat, 100 * 0.999)]
plt.xlim(xlimm)
plt.ylim(ylimm)
plt.contour(x_m, x_sigma2, f_joint.T, 6, colors=['b'])

# shadow plot for leg
acl = plt.plot(1000, 1000, color='b', lw=3)
plt.legend(['empirical scatter plot', 'analytical contour lines'])
plt.xlabel('$\hat{M}$')
plt.ylabel('$\hat{\Sigma}^2$')
plt.title('Sample mean-covariance joint distribution')
add_logo(fig4)

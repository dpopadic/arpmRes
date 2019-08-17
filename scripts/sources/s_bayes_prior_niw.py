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

# # s_bayes_prior_niw [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bayes_prior_niw&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFigNIWprior).

# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invwishart, norm, t

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_prior_niw-parameters)

mu_pri = 0.1  # prior expectation
sigma2_pri = 2.  # prior dispersion
t_pri = 7.  # confidence on mu_pri
v_pri = 5.  # confidence on sigma2_pri
k_ = 500  # number of grid points

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_prior_niw-implementation-step01): Compute the expectation and standard deviations of Sigma2 and M

# +
exp_sigma2 = invwishart.mean(v_pri, v_pri * sigma2_pri)  # expectation
std_sigma2 = np.sqrt(invwishart.var(v_pri, v_pri * sigma2_pri))  # std

exp_m = mu_pri  # expectation
std_m = np.sqrt((sigma2_pri / t_pri) * (v_pri / (v_pri - 2.)))  # std
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_prior_niw-implementation-step02): Compute the marginal pdf of Sigma2

y = np.linspace(0.1, exp_sigma2 + 3 * std_sigma2, k_)  # grid
f_sigma2 = invwishart.pdf(y, v_pri, v_pri * sigma2_pri)  # pdf

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_prior_niw-implementation-step03): Compute the marginal pdf of M

x = np.linspace(exp_m - 3 * std_m, exp_m + 3 * std_m, k_)  # grid
f_m = t.pdf((x - mu_pri) / np.sqrt(sigma2_pri / t_pri), v_pri) / \
      np.sqrt(sigma2_pri / t_pri)  # pdf

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bayes_prior_niw-implementation-step04): Compute the joint pdf of M and Sigma2

f_joint = np.zeros((k_, k_))
for k in range(k_):
    f_joint[k, :] = norm.pdf(x, mu_pri, np.sqrt(y[k] / t_pri)) * f_sigma2[k]

# ## Plots

# +
plt.style.use('arpm')

# pdf of Sigma2
fig = plt.figure()
plt.plot(y, f_sigma2, lw=1.5, color='b')
text = r'$\Sigma^{2} \sim InvWishart(\nu_{pri},  \nu_{pri}\sigma^2_{pri})$' + \
        '\n\n' + \
        r'$\nu_{pri}$=%3.0f, $\sigma_{pri}^2$=%3.1f' % (v_pri, sigma2_pri)
plt.text(0.7 * (y[-1] - y[0]) + y[0],
         0.7 * np.max(f_sigma2), text, color='b')
plt.xlabel('$\Sigma^2$')

add_logo(fig, location=1)
plt.tight_layout()

# pdf of M
fig = plt.figure()
plt.plot(x, f_m, lw=1.5, color='g')
text = r'$M \sim t (\nu_{pri},  \mu_{pri},  \sigma_{pri}^2 / t_{pri})$' + \
        '\n\n' + \
        r'$\nu_{pri}$=%3.0f, $t_{pri}$=%3.0f' % (v_pri, t_pri) + '\n' + \
        r'$\mu_{pri}$=%3.1f, $\sigma_{pri}^2$=%3.1f' % (mu_pri, sigma2_pri)
plt.text(0.7 * (x[-1] - x[0]) + x[0],
         0.7 * np.max(f_m), text, color='g')
plt.xlabel('$M$')

add_logo(fig, location=1)
plt.tight_layout()

# joint pdf
fig = plt.figure()
plt.contour(x, y, f_joint, 12, linewidths=1.5, colors='k')
plt.title('Joint pdf')
plt.xlabel('$M$')
plt.ylabel('$\Sigma^2$')

add_logo(fig)
plt.tight_layout()

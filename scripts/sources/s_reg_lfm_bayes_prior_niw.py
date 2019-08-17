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

# # s_reg_lfm_bayes_prior_niw [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_reg_lfm_bayes_prior_niw&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExFactNIWprior).

# +
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import invwishart, norm, t

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_prior_niw-parameters)

beta_pri = 0.5  # prior location parameter of the loadings
sigma2_pri = 3  # prior location parameter of the variance
sigma2_zpri = 2  # prior dispersion parameter of the loadings
t_pri = 3  # confidence on the prior loadings
v_pri = 10  # confidence on the prior variance
k_ = 500  # number of grid points

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_prior_niw-implementation-step01): Compute the expectation and standard deviations of Sigma2 and B

# +
exp_sigma2 = invwishart.mean(v_pri, v_pri * sigma2_pri)
std_sigma2 = np.sqrt(invwishart.var(v_pri, v_pri * sigma2_pri))

exp_beta = beta_pri
std_beta = np.sqrt(sigma2_pri / (sigma2_zpri * t_pri) * v_pri / (v_pri - 2.))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_prior_niw-implementation-step02): Compute the marginal pdf of Sigma2

s = np.linspace(0.1, exp_sigma2 + 3 * std_sigma2, k_)
f_sigma2 = invwishart.pdf(s, v_pri, v_pri * sigma2_pri)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_prior_niw-implementation-step03): Compute the marginal pdf of B

b = np.linspace(exp_beta - 3 * std_beta, exp_beta + 3 * std_beta, k_)
f_beta = t.pdf((b - beta_pri) / np.sqrt(sigma2_pri / (sigma2_zpri * t_pri)),
               v_pri) / np.sqrt(sigma2_pri / (sigma2_zpri * t_pri))

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_reg_lfm_bayes_prior_niw-implementation-step04): Compute the joint pdf of B and Sigma2

f_joint = np.zeros((k_, k_))
for k in range(k_):
    f_joint[:, k] = norm.pdf(b, beta_pri, np.sqrt(s[k] /
                             (sigma2_zpri * t_pri))) * f_sigma2[k]

# ## Plots

# +
plt.style.use('arpm')

# pdf of Sigma2
fig1 = plt.figure()
plt.plot(s, f_sigma2, lw=1.5, color='b')
text = r'$\Sigma^{2} \sim InvWishart(\nu_{pri},  \nu_{pri}\sigma^2_{pri})$' + \
        '\n\n' + \
        r'$\nu_{pri}$=%3.0f, $\sigma_{pri}^2$=%3.1f' % (v_pri, sigma2_pri)
plt.text(0.7 * (s[-1] - s[0]) + s[0],
         0.7 * np.max(f_sigma2), text, color='b')
plt.xlabel('$\Sigma^2$')

add_logo(fig1, location=1)

# pdf of M
fig2 = plt.figure()
plt.plot(b, f_beta, lw=1.5, color='g')

text = r'$B \sim t (\nu_{pri},\beta_{pri},\sigma_{pri}^2,' + \
        '(t_{pri}\sigma^2_{Z,pri})^{-1})$' + '\n\n' + \
        r'$\nu_{pri}$=%3.0f, $t_{pri}$=%3.0f' % (v_pri, t_pri) + '\n' + \
        r'$\beta_{pri}$=%3.1f, $\sigma_{pri}^2$=%3.1f, $\sigma_{Z, pri}^2$=%3.1f' % (beta_pri, sigma2_pri, sigma2_zpri)

plt.text(0.7 * (b[-1] - b[0]) + b[0],
         0.7 * np.max(f_beta), text, color='g')
plt.xlabel('$B$')

add_logo(fig2, location=1)

# joint pdf
fig3 = plt.figure()
ax = Axes3D(fig3)

x, s = np.meshgrid(b, s)
ax.plot_surface(b, s, f_joint.T)
ax.view_init(30, -120)
ax.set_title('joint pdf')
ax.set_xlabel('$B$')
ax.set_ylabel('$\Sigma^2$')

add_logo(fig3)

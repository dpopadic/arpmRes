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

# # s_entropy_view [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_entropy_view&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EntrpPool).

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

from arpym.statistics import meancov_sp
from arpym.views import min_rel_entropy_sp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-parameters)

# +
j_ = 10 ** 6  # number of scenarios
nu = 4  # degrees of freedom
mu = 0  # expectation of sample
sig = 1  # standard deviation of sample

mu_x_base = -2.2
sig_x_base = 1.3
sk_x_base = 4

c = 0.7    # confidence level
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-implementation-step01): Generate scenarios and uniform flexible probabilities of base distribution

x = (mu + sig * t.rvs(nu, size=(j_,)))
p_base_unif = np.ones((j_)) / j_  # base uniform flexible probabilities

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-implementation-step02): Compute base probabilities

# +
# Generate parameters specifying constraints for base distribution


def v(x):
    return np.array([x ** 3, x, x ** 2])  # view function


def mu_view_ineq(mu, sig, sk):
    return np.array([sk * sig ** 3 + 3 * mu * sig ** 2 + mu ** 3])


def mu_view_eq(mu, sig):
    return np.array([mu, mu ** 2 + sig ** 2])


z_ineq_base = - v(x)[:1]
mu_ineq_base = - mu_view_ineq(mu_x_base, sig_x_base, sk_x_base)

z_eq_base = v(x)[1:]
mu_view_eq_base = mu_view_eq(mu_x_base, sig_x_base)

p_base = min_rel_entropy_sp(p_base_unif, z_ineq_base, mu_ineq_base, z_eq_base,
                            mu_view_eq_base, normalize=False)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-implementation-step03): Compute updated probabilities

# +
# Generate parameters specifying constraints for updated distribution

z_ineq = v(x)[:1]
mu_ineq = mu_view_ineq(- mu_x_base, sig_x_base, - sk_x_base)

z_eq = v(x)[1:]
mu_view_eq = mu_view_eq(- mu_x_base, sig_x_base)

p_upd = min_rel_entropy_sp(p_base, z_ineq, mu_ineq, z_eq, mu_view_eq,
                           normalize=False)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-implementation-step04): Compute expectations, standard deviations and skewness of updated distribution

mu_upd, sig2_upd = meancov_sp(x, p_upd)
sig_upd = np.sqrt(sig2_upd)
sk_upd = ((x - mu_upd) ** 3) @ p_upd / sig_upd ** 3

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-implementation-step05): Compute confidence-weighted probabilities

p_c_add = c * p_upd + (1 - c) * p_base
p_c_mul = p_upd ** c * p_base ** (1 - c) /\
    np.sum(p_upd ** c * p_base ** (1 - c))

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_entropy_view-implementation-step06): Compute expectations, standard deviations and skewness of confidence-weighted distributions

# +
mu_c_add, sig2_c_add = meancov_sp(x, p_c_add)
sig_c_add = np.sqrt(sig2_c_add)
sk_add = ((x - mu_c_add) ** 3) @ p_c_add / sig_c_add ** 3

mu_c_mul, sig2_c_mul = meancov_sp(x, p_c_mul)
sig_c_mul = np.sqrt(sig2_c_mul)
sk_mul = ((x - mu_c_mul) ** 3) @ p_c_mul / sig_c_mul ** 3
# -

# ## Plots

# +
plt.style.use('arpm')
fig, ax = plt.subplots(4, 1)

xmin = -7
xmax = 7
ymin = -0.1
ymax = 0.65

# base distribution
plt.sca(ax[0])
f, xp = histogram_sp(x, p=p_base, k_=np.sqrt(j_))
plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.9, .9, .9], edgecolor='k')

sd_bar_base = np.linspace(mu_x_base - sig_x_base, mu_x_base + sig_x_base, 2)
plt.plot(sd_bar_base, [0, 0], 'b', lw=2, label='Standard deviation')
plt.plot(mu_x_base, 0, '.r', markersize=15, label='Expectation')
plt.title('Base distribution')

# updated distribution
plt.sca(ax[3])
f, xp = histogram_sp(x, p=p_upd, k_=np.sqrt(j_))
plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.9, .9, .9], edgecolor='k')

sd_bar_upd = np.linspace(mu_upd - sig_upd, mu_upd + sig_upd, 2)
plt.plot(sd_bar_upd, [0, 0], 'b', lw=2)
plt.plot(mu_upd, 0, '.r', markersize=15)
plt.title('Updated distribution')

# additive confidence-weighted distribution
plt.sca(ax[1])
f, xp = histogram_sp(x, p=p_c_add, k_=np.sqrt(j_))
plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.9, .9, .9], edgecolor='k')

sd_bar_p_c = np.linspace(mu_c_add - sig_c_add, mu_c_add + sig_c_add, 2)
plt.plot(sd_bar_p_c, [0, 0], 'b', lw=2)
plt.plot(mu_c_add, 0, '.r', markersize=15)
plt.title('Additive opinion pooling c = %d %%' % np.floor(c*100))

# multiplicative confidence-weighted distribution
plt.sca(ax[2])
f, xp = histogram_sp(x, p=p_c_mul, k_=np.sqrt(j_))
plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.9, .9, .9], edgecolor='k')

sd_bar_p_c_m = np.linspace(mu_c_mul - sig_c_mul, mu_c_mul + sig_c_mul, 2)
plt.plot(sd_bar_p_c_m, [0, 0], 'b', lw=2)
plt.plot(mu_c_mul, 0, '.r', markersize=15)
plt.title('Multiplicative opinion pooling c = %d %%' % np.floor(c*100))

for n in range(4):
    ax[n].set_yticks(np.linspace(0, 0.6, 4))
    ax[n].set_xlim([xmin, xmax])
    ax[n].set_ylim([ymin, ymax])
add_logo(fig, location=1)
plt.tight_layout()

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

# # s_location_stress_error [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_location_stress_error&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerStressErr).

# +
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import simulate_unif_in_ellips
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_location_stress_error-parameters)

k_ = 400  # cardinality of stress-test set
t_ = 15  # len of the time series
j_ = 5*10 ** 2  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_location_stress_error-implementation-step01): Set the stress test set for the true parameters

# +
# generate uniform on unit circle
unif, _, _ = simulate_unif_in_ellips(np.array([2, 2]), np.identity(2),
                                     int(k_/2))
mu = unif[:, 0]
sigma2 = unif[:, 1]
# ensemble error
m = 2*np.log(mu) - 0.5*np.log(sigma2 + mu ** 2)
s2 = 1.2*np.log((sigma2 / mu ** 2) + 1)

location = np.r_[mu, m]
dispersion = np.r_[sigma2, s2]

# vector of true expectations
expectation = np.r_[mu, mu]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_location_stress_error-implementation-step02): Generate scenarios and compute the error for each estimator

# +
m_hat = np.zeros((j_, k_))
pi_hat = np.zeros((j_, k_))
k_hat = np.zeros((j_, k_))
er_m = np.zeros(k_)
er_pi = np.zeros(k_)
er_k = np.zeros(k_)

for k in range(k_):
    # generate scenarios
    if k <= int(k_ / 2)-1:
        # normal simulations
        i_thetak = stats.norm.rvs(location[k], np.sqrt(dispersion[k]),
                                  size=[j_, t_])
    else:
        # lognormal simulations
        i_thetak = stats.lognorm.rvs(np.sqrt(dispersion[k]),
                                     scale=np.exp(location[k]), size=[j_, t_])
    # sample mean estimator
    m_hat[:, k] = np.mean(i_thetak, axis=1)  # simulations
    l_m = (m_hat[:, k]-expectation[k]) ** 2  # loss
    er_m[k] = np.mean(l_m)  # error
    # product estimator
    pi_hat[:, k] = i_thetak[:, 0] * i_thetak[:, -1]  # simulations
    l_pi = (pi_hat[:, k]-expectation[k]) ** 2  # loss
    er_pi[k] = np.mean(l_pi)  # error
    # constant estimator
    k_hat[:, k] = 3*np.ones(j_)  # simulations
    l_k = (k_hat[:, k]-expectation[k]) ** 2  # loss
    er_k[k] = np.mean(l_k)  # error
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_location_stress_error-implementation-step03): Compute robust and ensemble error for each estimator

# +
# robust errors
er_rob_m, i_m = np.max(er_m), np.argmax(er_m)
er_rob_pi, i_pi = np.max(er_pi), np.argmax(er_pi)
er_rob_k, i_k = np.max(er_k), np.argmax(er_k)

# ensemble errors
er_ens_m = np.mean(er_m)
er_ens_pi = np.mean(er_pi)
er_ens_k = np.mean(er_k)
# -

# ## Save database

# +
output = {'j_': pd.Series(k_),
          'k_': pd.Series(k_),
          'm_hat': pd.Series(m_hat.reshape((j_*k_,))),
          'expectation': pd.Series(expectation),
          'er_rob_m': pd.Series(er_rob_m),
          'er_ens_m': pd.Series(er_ens_m)}

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_stress_error.csv',
          index=None)
# -

# ## Plots

# +
plt.style.use('arpm')

# preliminary computations
p = 0.025
x_min = -3
x_max = 10
y_min = 0
y_max = 1.2
# compute pdf's
x_vec = np.arange(x_min, x_max+0.05, 0.05)
if i_m > k_ / 2:
    pdf_m = stats.lognorm.pdf(x_vec, np.sqrt(dispersion[i_m]),
                              scale=np.exp(location[i_m]))
else:
    pdf_m = stats.norm.pdf(x_vec, location[i_m], np.sqrt(dispersion[i_m]))

if i_pi > k_ / 2:
    pdf_pi = stats.lognorm.pdf(x_vec, np.sqrt(dispersion[i_pi]),
                               scale=np.exp(location[i_pi]))
else:
    pdf_pi = stats.norm.pdf(x_vec, location[i_pi], np.sqrt(dispersion[i_pi]))

if i_k > k_ / 2:
    pdf_k = stats.lognorm.pdf(x_vec, np.sqrt(dispersion[i_k]),
                              scale=np.exp(location[i_k]))
else:
    pdf_k = stats.norm.pdf(x_vec, location[i_k], np.sqrt(dispersion[i_k]))

# initialize strings
epsi_string = '$\epsilon$'
dist_string = {}
m_hat_string = {}
ss2_string = {}
for k in range(int(k_ / 2)):
    dist_string[k] = 'N'
    dist_string[k+k_ / 2] = 'LogN'
    m_hat_string[k] = '$\mu$'
    m_hat_string[k+int(k_ / 2)] = 'm'
    ss2_string[k] = '$\sigma^{2}$'
    ss2_string[k+int(k_ / 2)] = '$s^{2}$'

# color settings
orange = [1, 0.4, 0]
grey = [0.4, 0.4, 0.4]
blue = [0, 0.4, 1]
red = [0.9, 0.3, 0.1]

fig, ax = plt.subplots(2, 2)

# pdf plot
ax1 = plt.subplot(2, 1, 1)

plt.plot(x_vec, pdf_m, color=grey, lw=1.5)
plt.plot(x_vec, pdf_pi, color=red, lw=1.5)
plt.plot(x_vec, pdf_k, color=blue, lw=1.5)
plt.xlabel('$\epsilon$')
plt.title('TRUE UNKNOWN DISTRIBUTION')
plt.xticks(np.arange(-5, x_max+1))
plt.ylim([y_min, y_max])
plt.yticks([])
m_string =\
    'Sample mean robust error(%s$_{t}\sim$%s(%s=%3.2f,%s=%3.2f)): %3.2f' % \
    (epsi_string, dist_string[i_m], m_hat_string[i_m], location[i_m],
     ss2_string[i_m], dispersion[i_m], er_rob_m)
plt.text(x_max, (0.725 + p)*y_max, m_string, color=grey,
         horizontalalignment='right')
pi_string = \
'First - last product robust error( %s$_{t}\sim$%s(%s=%3.2f,%s=%3.2f)): %3.2f'\
   % (epsi_string, dist_string[i_pi], m_hat_string[i_pi], location[i_pi],
      ss2_string[i_pi], dispersion[i_pi], er_rob_pi)
plt.text(x_max, (0.475 + p)*y_max, pi_string, color=red,
         horizontalalignment='right')
k_string = \
    'Constant robust error( % s$_{t}\sim$%s(%s=%3.2f,%s=%3.2f)): %3.2f' % \
    (epsi_string, dist_string[i_k], m_hat_string[i_k], location[i_k],
     ss2_string[i_k], dispersion[i_k], er_rob_k)
plt.text(x_max, (0.6 + p)*y_max, k_string, color='b',
         horizontalalignment='right')
ax1.set_xlim([-0.25, x_max])

# parameters plot
plt.sca(ax[1, 0])
plt.scatter(mu, sigma2, 3, 'k', '.')
plt.xlabel('$\mu$')
plt.ylabel('$\sigma^2$')
plt.axis('equal')
plt.xlim([np.min(mu), np.max(mu)])
plt.ylim([0, 1.1*np.max(sigma2)])
plt.title('Normal parameters')

plt.sca(ax[1, 1])
plt.scatter(m, s2, 3, 'k', '.')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('$s^2$')
plt.xlim([np.min(m), np.max(m)])
plt.ylim([0, 1.1*np.max(s2)])
plt.title('LogNormal parameters')
add_logo(fig, location=1, size_frac_x=1/8)
plt.tight_layout()

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

# # s_min_vol_ellips [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_min_vol_ellips&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-hfpellips-exercise).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools import plot_ellipse, mahalanobis_dist, add_logo
from arpym.statistics import simulate_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_min_vol_ellips-parameters)

j_ = 5000
mu = np.array([0, 0])  # expectation
rho = .6  # correlation
sigma2 = np.array([[1, rho], [rho, 1]])  # covariance

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_min_vol_ellips-implementation-step01): Generate j_normal scenarios

x = simulate_normal(mu, sigma2, j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_min_vol_ellips-implementation-step02): Rescale the covariance matrix

n_ = sigma2.shape[0]
sigma2_rescaled = n_ * sigma2

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_min_vol_ellips-implementation-step03): Generate location and dispersion to satisfy Mah distance constraint

# +
m = mu + np.random.rand(2)
a = np.random.rand(2, 2)
s2 = a @ a.T  # generate symmetric covariance matrix
mah_dist2 = np.zeros(j_)

for j in range(j_):
    mah_dist2[j] = (mahalanobis_dist(x[[j], :], m, s2))**2
r2 = np.mean(mah_dist2)  # average square Mahalanobis distance
s2 = s2 * r2
# -

# ## Plot

plt.style.use('arpm')
grey = [.5, .5, .5]
fig = plt.figure()
plt.plot([], [], color='r', lw=2)  # dummy plot for legend
plt.plot([], [], color='b', lw=2)  # dummy plot for legend
plot_ellipse(mu, sigma2_rescaled, r=1, color='r', line_width=2)
plot_ellipse(m, s2, r=1, color='b', line_width=2)
plt.scatter(x[:, 0], x[:, 1], s=5, color=grey)
plt.legend(('expectation-(rescaled)covariance ellipsoid',
           'generic ellipsoid (expected square Mah. distance = 1)'))
add_logo(fig)
plt.tight_layout()

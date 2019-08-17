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

# # s_encoding [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_encoding&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_encoding).

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_encoding-parameters)

n_samples = 2000  # number of samples
mu_z = np.zeros(2)  # expectation
sigma2_z = np.array([[1, 0], [0, 1]])  # covariance

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_encoding-implementation-step01): Define features and target variables

# +

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)


def sigf(z1, z2):
    return 0  # np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))


z = np.random.multivariate_normal(mu_z, sigma2_z, n_samples)

x = muf(z[:, 0], z[:, 1]) +\
       sigf(z[:, 0], z[:, 1]) * np.random.randn(n_samples)

q_z1_15 = np.percentile(z[:, 0], 1/5*100)
q_z1_25 = np.percentile(z[:, 0], 2/5*100)
q_z1_35 = np.percentile(z[:, 0], 3/5*100)
q_z1_45 = np.percentile(z[:, 0], 4/5*100)
q_z1 = [q_z1_15, q_z1_25, q_z1_35, q_z1_45]
c_z1 = len(q_z1)+1

q_z2_16 = np.percentile(z[:, 1], 1/6*100)
q_z2_26 = np.percentile(z[:, 1], 2/6*100)
q_z2_36 = np.percentile(z[:, 1], 3/6*100)
q_z2_46 = np.percentile(z[:, 1], 4/6*100)
q_z2_56 = np.percentile(z[:, 1], 5/6*100)
q_z2 = [q_z2_16, q_z2_26, q_z2_36, q_z2_46, q_z2_56]
c_z2 = len(q_z2)+1

z1 = np.ones(n_samples)
z1[z[:, 0] <= q_z1[0]] = 0
z1[np.logical_and(z[:, 0] > q_z1[0], z[:, 0] <= q_z1[1])] = 1
z1[np.logical_and(z[:, 0] > q_z1[1], z[:, 0] <= q_z1[2])] = 2
z1[np.logical_and(z[:, 0] > q_z1[2], z[:, 0] <= q_z1[3])] = 3
z1[z[:, 0] > q_z1[3]] = 4

z2 = np.ones(n_samples)
z2[z[:, 1] <= q_z2[0]] = 0
z2[np.logical_and(z[:, 1] > q_z2[0], z[:, 1] <= q_z2[1])] = 1
z2[np.logical_and(z[:, 1] > q_z2[1], z[:, 1] <= q_z2[2])] = 2
z2[np.logical_and(z[:, 1] > q_z2[2], z[:, 1] <= q_z2[3])] = 3
z2[np.logical_and(z[:, 1] > q_z2[3], z[:, 1] <= q_z2[4])] = 4
z2[z[:, 1] > q_z2[4]] = 5

z = np.c_[z1, z2]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_encoding-implementation-step02): Encode categorical variables

enc = OneHotEncoder()
z_cat = enc.fit_transform(z).toarray()

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_encoding-implementation-step03): Fit a regression model turning on encoded features one by one and compute error

# +
error = np.zeros(z_cat.shape[1]+1)

for l in range(z_cat.shape[1]):
    reg = linear_model.LinearRegression()
    x_hat = reg.fit(z_cat[:, :l+1], x).predict(z_cat[:, :l+1])
    error[l] = np.mean((x-x_hat)**2)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_encoding-implementation-step04): Add interactions and compute error

# +
z_cat_inter = np.zeros((n_samples, c_z1*c_z2))
k = 0
for k1 in range(c_z1):
    for k2 in range(c_z2):
        z_cat_inter[:, k] = z_cat[:, k1]*z_cat[:, c_z1+k2]
        k = k+1


x_hat = reg.fit(z_cat_inter, x).predict(z_cat_inter)
error[-1] = np.mean((x-x_hat)**2)
# -

# ## Plots

# +
plt.style.use('arpm')

lightblue = [0.2, 0.6, 1]
lightgreen = [0.6, 0.8, 0]

fig = plt.figure()

# Error
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
ax1.plot(1+np.arange(error.shape[0]), error, color='k')
ax1.set_xticks(1+np.arange(error.shape[0]))
ax1.set_xticklabels(['1 feat.', '2 feats.', '3 feats.', '4 feats.', '5 feats.',
                     '6 feats.', '7 feats.', '8 feats.', '9 feats.',
                     '10 feats.', '11 feats.', '2° ord. inter.'])
ax1.set_title('Error', fontweight='bold')
ax1.grid(False)

# Data
ax2 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2, projection='3d')
ax2.scatter3D(z[:, 0], z[:, 1], x, s=10, color=lightblue, alpha=1,
              label='$(Z_1, Z_2, X)$')
ax2.set_xlabel('$Z_1$')
ax2.set_xticks(np.arange(c_z1))
ax2.set_xticklabels(['1a', '1b', '1c', '1d', '1e'])
ax2.set_ylabel('$Z_2$')
ax2.set_yticks(np.arange(c_z2))
ax2.set_yticklabels(['2a', '2b', '2c', '2d', '2e', '2f'])
ax2.set_zlabel('$X$')
ax2.set_title('$(Z_1,Z_2,X)$', fontweight='bold')
# ax.legend()

# Fitted data
ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=2, projection='3d')
ax3.scatter3D(z[:, 0], z[:, 1], x_hat, s=15, color=lightgreen, alpha=1,
              label='$(Z_1,Z_2, \hat{X})$')
ax3.set_xlabel('$Z_1$')
ax3.set_xticks(np.arange(c_z1))
ax3.set_xticklabels(['1a', '1b', '1c', '1d', '1e'])
ax3.set_ylabel('$Z_2$')
ax3.set_yticks(np.arange(c_z2))
ax3.set_yticklabels(['2a', '2b', '2c', '2d', '2e', '2f'])
ax3.set_zlabel('$\hat{X}$')
ax3.set_title('$(Z_1,Z_2,\hat{X})$', fontweight='bold')
# ax.legend()
add_logo(fig, size_frac_x=1/8)
plt.tight_layout()

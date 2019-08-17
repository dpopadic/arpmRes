#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_bias_vs_variance_normal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bias_vs_variance_normal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_normal).

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_normal-parameters)

n_sample = 2000  # number of samples
mu_z = np.zeros(2)  # expectation
sigma2_z = np.array([[1, 0], [0, 1]])  # covariance
pol_degree = 10  # maximum degree of polynomials considered
j_ = 1000  # simulations of out-of-sample error

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_normal-implementation-step00): Initialize variables

err_in = np.zeros(pol_degree)
err_out = np.zeros((j_, pol_degree))
err_out_med = np.zeros(pol_degree)
err_out_iqr = np.zeros(pol_degree)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_normal-implementation-step01): Define features and target variables

# +

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)


def sigf(z1, z2):
    return np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))


z_in = np.random.multivariate_normal(mu_z, sigma2_z, n_sample)
x_in = muf(z_in[:, 0], z_in[:, 1]) +\
       sigf(z_in[:, 0], z_in[:, 1]) * np.random.randn(n_sample)


for deg in np.arange(pol_degree):

    # Step 2: Construct interactions in-sample

    poly = PolynomialFeatures(degree=deg+1, include_bias=False)
    z_inter_in = poly.fit_transform(z_in)

    # Step 3: Fit conditional expectation through regression in-sample

    reg = linear_model.LinearRegression()
    exp_in_sample = reg.fit(z_inter_in, x_in).predict(z_inter_in)

    # Step 4: Compute in-sample error

    err_in[deg] = np.mean((x_in-exp_in_sample)**2)

    # Step 5: Compute distribution of out-of-sample error

    for j in range(j_):

        # generate out-of-sample
        z_out = np.random.multivariate_normal(mu_z, sigma2_z, n_sample)
        x_out = muf(z_out[:, 0], z_out[:, 1]) +\
            sigf(z_out[:, 0], z_out[:, 1]) * np.random.randn(n_sample)
        poly = PolynomialFeatures(degree=deg+1, include_bias=False)
        z_inter_out = poly.fit_transform(z_out)

        # error
        exp_out_sample = reg.predict(z_inter_out)
        err_out[j, deg] = np.mean((x_out-exp_out_sample)**2)

    err_out_med[deg] = np.median(err_out[:, deg])
    err_out_iqr[deg] = np.percentile(err_out[:, deg], 75) -\
        np.percentile(err_out[:, deg], 25)
# -

# ## Plots

# +
plt.style.use('arpm')

idxx0 = np.where(np.abs(z_in[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(z_in[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)
lightblue = [0.2, 0.6, 1]
lightgreen = [0.6, 0.8, 0]

fig = plt.figure()

# Error
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
insamplot = ax1.plot(np.arange(pol_degree)+1, err_in, color='k')
ax1.set_ylabel('In-sample error', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.set_xticks(np.arange(pol_degree)+1)
ax12 = ax1.twinx()
outsamplot = ax12.plot(np.arange(pol_degree)+1, err_out_med, color='r',
                       lw=1.15)
ax12.tick_params(axis='y', colors='r')
ax12.set_ylabel('Out-of-sample error', color='r')
ax1.set_xlabel('Degree of the polynomial')
plt.xlim([0, pol_degree + 1])
ax1.set_title('In-sample vs out-of-sample errors as ' +
              'function of polynomial degree', fontweight='bold')
ax1.grid(False)
ax12.grid(False)

# Conditional expectation surface
ax2 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2, projection='3d')
step = 0.01
zz1, zz2 = np.meshgrid(np.arange(-2, 2, step), np.arange(-2, 2, step))
ax2.plot_surface(zz1, zz2, muf(zz1, zz2), color=lightblue, alpha=0.7,
                 label='$\mu(z_1, z_2)$')

ax2.scatter3D(z_in[idxx, 0], z_in[idxx, 1],
              x_in[idxx], s=10, color=lightblue, alpha=1,
              label='$(Z_1, Z_2, X)$')
ax2.set_xlabel('$Z_1$')
ax2.set_ylabel('$Z_2$')
ax2.set_zlabel('$X$')
ax2.set_title('Conditional expectation surface', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])

# Fitted surface
ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=2, projection='3d')
step = 0.01
zz1, zz2 = np.meshgrid(np.arange(-2, 2, step), np.arange(-2, 2, step))
zz = poly.fit_transform(np.c_[zz1.ravel(), zz2.ravel()])
xx = reg.predict(zz)
ax3.plot_surface(zz1, zz2, xx.reshape((zz1.shape)), color=lightgreen,
                 alpha=0.7, label='Fitted surface')

ax3.scatter3D(z_in[idxx, 0], z_in[idxx, 1],
              reg.predict(z_inter_in)[idxx], s=10, color=lightgreen,
              alpha=1, label='$(Z_1,Z_2, \hat{X})$')
ax3.set_xlabel('$Z_1$')
ax3.set_ylabel('$Z_2$')
ax3.set_zlabel('$\hat{X}$')
ax3.set_title('Fitted surface', fontweight='bold')
ax3.set_xlim([-2, 2])
ax3.set_ylim([-2, 2])

add_logo(fig, axis=ax1)
plt.tight_layout()

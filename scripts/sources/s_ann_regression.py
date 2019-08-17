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

# # s_ann_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ann_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_ann_regression).

# +
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ann_regression-parameters)

max_l_ = 4
n_sample = 1000
mu_z = np.zeros(2)  # expectation
sigma2_z = np.array([[1, 0], [0, 1]])  # covariance

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ann_regression-implementation-step01): Define features and target variables

# +

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)


def sigf(z1, z2):
    return np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))


z = np.random.multivariate_normal(mu_z, sigma2_z, n_sample)
x = muf(z[:, 0], z[:, 1]) +\
       sigf(z[:, 0], z[:, 1]) * np.random.randn(n_sample)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ann_regression-implementation-step02): Artificial neural network

err_in = np.zeros(max_l_)
for i in np.arange(1, max_l_+1):
    hidlay = tuple([100 for i in range(i)])
    ann_reg = MLPRegressor(hidden_layer_sizes = hidlay, solver = 'lbfgs')
    err_in[i-1] = np.mean((x-ann_reg.fit(z, x).predict(z))**2)
print(hidlay)
print(err_in)

# ## Plots

# +
plt.style.use('arpm')

idxx0 = np.where(np.abs(z[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(z[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)
lightblue = [0.2, 0.6, 1]
lightgreen = [0.6, 0.8, 0]

fig = plt.figure()

# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.06

z_1_min = z[:, 0].min()
z_1_max = z[:, 0].max()
z_2_min = z[:, 1].min()
z_2_max = z[:, 1].max()
zz1, zz2 = np.meshgrid(np.arange(z_1_min, z_1_max, plot_step),
                       np.arange(z_2_min, z_2_max, plot_step))

# Error
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
insamplot = ax1.plot(np.arange(max_l_)+1, err_in, color='k')
ax1.set_ylabel('In-sample error', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.set_xlabel('Number of layers')
plt.xlim([0, max_l_ + 1])
ax1.set_title('In-sample error as function of number of layers',
              fontweight='bold')
ax1.grid(False)

# Conditional expectation surface
ax2 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2,
                       projection='3d')
step = 0.01
zz1, zz2 = np.meshgrid(np.arange(-2, 2, step), np.arange(-2, 2, step))
ax2.plot_surface(zz1, zz2, muf(zz1, zz2), color=lightblue, alpha=0.7,
                 label='$\mu(z_1, z_2)$')

ax2.scatter3D(z[idxx, 0], z[idxx, 1],
              x[idxx], s=10, color=lightblue, alpha=1,
              label='$(Z_1, Z_2, X)$')
ax2.set_xlabel('$Z_1$')
ax2.set_ylabel('$Z_2$')
ax2.set_zlabel('$X$')
ax2.set_title('Conditional expectation surface', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
# ax.legend()

# Fitted surface
ax3 = plt.subplot2grid((3, 4), (1, 2), rowspan=2, colspan=2, projection='3d')
x_plot = ann_reg.predict(np.c_[zz1.ravel(), zz2.ravel()])
x_plot = x_plot.reshape(zz1.shape)
ax3.plot_surface(zz1, zz2, x_plot, alpha=0.5, color=lightgreen)
ax3.scatter3D(z[idxx, 0], z[idxx, 1],
              ann_reg.predict(z[idxx, :]), s=10,
              alpha=1, color=lightgreen)
ax3.set_xlabel('$Z_1$')
ax3.set_ylabel('$Z_2$')
ax3.set_zlabel('$\hat{X}$')
plt.title('Fitted surface; \n n_sample = %1i; ' % n_sample +
          'Layers = %1i; ' % max_l_,
          fontweight='bold')

add_logo(fig, size_frac_x=1/8)
plt.tight_layout()

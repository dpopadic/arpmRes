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

# # s_logn_mean_lin_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_logn_mean_lin_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_lin_regression).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arpym.tools import add_logo
from arpym.statistics import simulate_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_lin_regression-parameters)

# +
mu_xz = [0, 0]  # location parameter
# dispersion parameters
rho_xz = 0.8
sig_x = 0.92
sig_z = 0.85
j_ = 10**4  # number of simulations


# Arbitrary linear predictor
def chi_arb(var):
    return 1/3*var+1
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_lin_regression-implementation-step01): Generate samples

# +
sig2_xz = np.array([[sig_x**2, rho_xz*sig_x*sig_z],
                    [rho_xz*sig_x*sig_z, sig_z**2]])
# jointly lognormal samples
x, z = np.exp(simulate_normal(mu_xz, sig2_xz, j_).T)


# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_lin_regression-implementation-step02): Compute optimal/arbitrary predictor, prediction and residuals

beta = np.exp(mu_xz[0] - mu_xz[1] + 0.5*(sig_x**2 - sig_z**2)) *\
        (np.exp(rho_xz*sig_x*sig_z) - 1)/(np.exp(sig_z**2) - 1)
alpha = np.exp(mu_xz[0] + 0.5*sig_x**2) - beta*np.exp(mu_xz[1] + 0.5*sig_z**2)


# Best linear predictor
def chi_alpha_beta(var):
    return alpha + beta*var


xhat = chi_alpha_beta(z)
u = x-xhat
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_lin_regression-implementation-step03): Expectation and covariance of (Xhat, U)

# Expectation and covariance of lognormal variable
exp_x_z = np.exp(mu_xz + 0.5*np.diag(sig2_xz))
cv_x_z = np.diag(exp_x_z)@(np.exp(sig2_xz) - np.ones((2, 2)))@np.diag(exp_x_z)
# expectation and covariance of (Xhat,X)
a = np.array([alpha, -alpha]).reshape(-1, 1)
b = np.array([[0, beta], [1, -beta]])
mu_xhat_u = a + b@np.reshape(exp_x_z, (-1, 1))
sig2_xhat_u = b@cv_x_z@b.T

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_lin_regression-implementation-step04): expectation-covariance ellipsoid computations

# +
# matrix decomposition
lambda2, e = np.linalg.eig(sig2_xhat_u)
lambda2, order = np.sort(lambda2), np.argsort(lambda2)
e = e[:, order]
diag_lambda = np.diagflat(np.sqrt(lambda2))

# expectation-covariance ellipsoid computations
theta = np.linspace(0, 2*np.pi, 200)  # angle
y = [np.cos(theta), np.sin(theta)]  # circle parametrization
axes_points = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
ellipse = mu_xhat_u + e@diag_lambda@y
axes_points_transformed = mu_xhat_u + e@diag_lambda@axes_points
# -

# ## Plots

# +
# Compute the rectangles vertices and edges

# select 2 simulations closest to the points below to display squared distance from
anchor_point_1 = [0.2, 0.7]
anchor_point_2 = [3.2, 3]
index1 = np.argmin(np.sum(([x, z]-np.reshape(anchor_point_1, (2, 1)))**2,
                          axis=0))
index2 = np.argmin(np.sum(([x, z]-np.reshape(anchor_point_2, (2, 1)))**2,
                          axis=0))
chi_z_val_select1 = chi_alpha_beta(z[index1])
square1_edge = chi_z_val_select1-x[index1]
chi_z_val_select2 = chi_alpha_beta(z[index2])
square2_edge = chi_z_val_select2-x[index2]
arb_fun_select1 = chi_arb(z[index1])
square3_edge = arb_fun_select1-x[index1]
arb_fun_select2 = chi_arb(z[index2])
square4_edge = arb_fun_select2-x[index2]

n_grid = 500
x_grid = np.linspace(10**-4, 2*max(np.percentile(x, 95), np.percentile(z, 95)),
                     n_grid)

plt.style.use('arpm')
# colors
teal = [0.2344, 0.582, 0.5664]
light_teal = [0.2773, 0.7031, 0.6836]
light_green_1 = [0.8398, 0.9141, 0.8125]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.6, 0.6, 0.6]
orange = [0.94, 0.35, 0]
black = [0, 0, 0]
blue = [0, 0, 1]
colf = [0, 0.5412, 0.9020]
trans = 0.2
markersize = 4
j_plot = 1000  # number of plotted simulations


def isinteger(x):
    return x[np.equal(np.mod(x, 1), 0)]


def logn_pdf(x, mu, sig):
    return 1/(x*(sig*np.sqrt(2*np.pi))) *\
            np.exp(-((np.log(x)-mu)**2/(2*sig**2)))


fig = plt.figure(dpi=72)
upper_limit = max(np.percentile(x, 95), np.percentile(z, 95))

ax1 = plt.subplot2grid((10, 16), (0, 3), colspan=12, rowspan=6)
ax1.set_aspect('equal')
ax1.set_xlim([0, 1.53*upper_limit])
ax1.set_ylim([0, upper_limit])
ax1.set_xticks(isinteger(ax1.get_xticks()))
ax1.set_yticks(isinteger(ax1.get_yticks()))
ax1.tick_params(axis='both', which='major', pad=-20)
ax1.set_facecolor('none')
ax1.grid(False)
ax1.set_xlabel('$Z$', position=(1, 0), fontdict={'size': 16}, labelpad=-30)
ax1.set_ylabel('$X$', position=(0, 1), fontdict={'size': 16}, labelpad=-20)
xlimm1 = ax1.get_xlim()
ax1.scatter(z[:j_plot], x[:j_plot], s=markersize, c=[light_grey])
l5, = ax1.plot(x_grid, chi_alpha_beta(x_grid), c=orange, lw=2)
l6, = ax1.plot(x_grid, chi_arb(x_grid), c='b', lw=2)
ax1.scatter([z[index1], z[index2], z[index1], z[index2], z[index1], z[index2]],
            [x[index1], x[index2], chi_z_val_select1, chi_z_val_select2,
             arb_fun_select1, arb_fun_select2],
            s=markersize*30, color=[black, black, orange, orange, blue, blue])
r1 = Rectangle((z[index1], x[index1]), square1_edge, square1_edge, fill=True,
               facecolor=orange, ec=orange, alpha=2*trans)
r2 = Rectangle((z[index2], x[index2]), square2_edge, square2_edge, fill=True,
               facecolor=orange, ec=orange, alpha=2*trans)
r3 = Rectangle((z[index1], x[index1]), square3_edge, square3_edge, fill=True,
               facecolor='b', ec='b', alpha=trans)
r4 = Rectangle((z[index2], x[index2]), square4_edge, square4_edge, fill=True,
               facecolor='b', ec='b', alpha=trans)
[ax1.add_patch(patch) for patch in [r1, r2, r3, r4]]
# text box xhat
x_limits = ax1.get_xlim()
y_limits = ax1.get_ylim()
ax1.annotate('$\hat{x}=\chi(z)$', fontsize=20, color=orange,
             horizontalalignment='center',
             xy=(0.46*(x_limits[1]-x_limits[0]),
                 chi_alpha_beta(0.46*(x_limits[1]-x_limits[0]))),
             xytext=(0.46*(x_limits[1]-x_limits[0]), 0.8*y_limits[1]),
             arrowprops=dict(facecolor=orange, ec='none', shrink=0.05))
plt.box(False)

ax2 = plt.subplot2grid((10, 16), (0, 1), colspan=2, rowspan=6, sharey=ax1)
ax2.set_ylim([0, upper_limit])
ax2.set_xticks([])
ax2.yaxis.set_ticks_position('none')
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis='y', which='major', pad=20)
ax2.set_facecolor('none')
ax2.grid(False)
ax2.invert_xaxis()
ax2.hist(x, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
plt.box(False)

ax3 = plt.subplot2grid((10, 16), (6, 3), colspan=12, rowspan=1, sharex=ax1)
ax3.set_xlim([0, 1.53*upper_limit])
ax3.set_xticks(ax3.get_xticks()[1:])
ax3.xaxis.set_ticks_position('none')
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_yticks([])
ax3.set_facecolor('none')
ax3.grid(False)
ax3.invert_yaxis()
ax3.hist(z, bins='auto', density=True, facecolor=light_green_2,
         ec=light_green_2)
plt.box(False)

ax4 = plt.subplot2grid((20, 16), (14, 1), colspan=7, rowspan=5)
ax4.set_facecolor('none')
ax4.grid(False)
ax4.set_xlabel('$\chi(Z)$', position=(0.9, 0), fontdict={'size': 16},
               labelpad=-30)
ax4.set_ylabel('$X$', position=(0, 1), fontdict={'size': 16}, labelpad=-30)
ax4.tick_params(axis='both', which='major', pad=-12)
ax4.set_aspect('equal')
ax4.set_xlim([0, 2*upper_limit])
ax4.set_ylim([0, upper_limit])
ax4.set_xticks(isinteger(ax4.get_xticks())[1:])
ax4.set_yticks(isinteger(ax4.get_yticks()))
ax4.scatter(xhat[:j_plot], x[:j_plot], s=markersize, c=[light_grey])
plt.box(False)

ax5 = plt.subplot2grid((20, 16), (14, 0), colspan=1, rowspan=5, sharey=ax4)
ax5.set_xticks([])
ax5.set_facecolor('none')
ax5.grid(False)
ax5.yaxis.set_ticks_position('none')
plt.setp(ax5.get_yticklabels(), visible=False)
ax5.set_ylim([0, upper_limit])
ax5.invert_xaxis()
scale = (x_grid-alpha)/beta
ax5.hist(x, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
ax5.plot(logn_pdf(scale[scale > 0], mu_xz[1], abs(sig_z))/abs(beta),
         x_grid[scale > 0], c=light_green_1, lw=2)
plt.box(False)

ax6 = plt.subplot2grid((20, 16), (19, 1), colspan=7, rowspan=1, sharex=ax4)
ax6.set_yticks([])
ax6.set_facecolor('none')
ax6.grid(False)
ax6.xaxis.set_ticks_position('none')
plt.setp(ax6.get_xticklabels(), visible=False)
ax6.set_xlim([0, 2*upper_limit])
ax6.invert_yaxis()
ax6.plot(x_grid[scale > 0], logn_pdf(scale[scale > 0], mu_xz[1],
         abs(sig_z))/abs(beta), c=light_green_1, lw=2)

plt.box(False)

ax7 = plt.subplot2grid((20, 16), (14, 9), colspan=7, rowspan=5)
ax7.set_facecolor('none')
ax7.grid(False)
ax7.set_xlabel('$\chi(Z)$', position=(0.9, 0), fontdict={'size': 16},
               labelpad=-30)
ax7.set_ylabel('$U$', position=(0, 1), fontdict={'size': 16}, labelpad=-30)
ax7.tick_params(axis='both', which='major', pad=-12)
ax7.set_aspect('equal')
ax7.set_xlim([0, 2*upper_limit])
ax7.set_ylim([-upper_limit/2, upper_limit/2])
ax7.set_xticks(isinteger(ax7.get_xticks())[1:])
ax7.set_yticks(isinteger(ax7.get_yticks()))
ax7.scatter(xhat[:j_plot], u[:j_plot], s=markersize, c=[light_grey])
ax7.plot(ellipse[0], ellipse[1],
         axes_points_transformed[0, 0:2], axes_points_transformed[1, 0:2],
         axes_points_transformed[0, 2:], axes_points_transformed[1, 2:],
         c='k', lw=1)
plt.box(False)

ax8 = plt.subplot2grid((20, 16), (14, 8), colspan=1, rowspan=5, sharey=ax7)
ax8.set_xticks([])
ax8.set_facecolor('none')
ax8.grid(False)
ax8.yaxis.set_ticks_position('none')
plt.setp(ax8.get_yticklabels(), visible=False)
ax8.set_ylim([-upper_limit/2, upper_limit/2])
ax8.invert_xaxis()
ax8.hist(u, bins='auto', density=True, facecolor=colf, ec=colf,
         orientation='horizontal')
plt.box(False)

ax9 = plt.subplot2grid((20, 16), (19, 9), colspan=7, rowspan=1, sharex=ax7)
ax9.set_yticks([])
ax9.set_facecolor('none')
ax9.grid(False)
ax9.xaxis.set_ticks_position('none')
plt.setp(ax9.get_xticklabels(), visible=False)
ax9.set_xlim([0, 2*upper_limit])
ax9.invert_yaxis()
plt.box(False)

l3, = ax9.plot(x_grid[scale > 0], logn_pdf(scale[scale > 0], mu_xz[1],
               abs(sig_z))/abs(beta), c=light_green_1, lw=2)

l1 = Rectangle((0, 0), 1, 1, color=light_green_2, ec='none')
l2 = Rectangle((0, 0), 1, 1, color=teal, ec='none')
l4 = Rectangle((0, 0), 1, 1, color=colf, ec='none')
fig.legend((l1, l2, l3, l4, l5, l6),
           ('Input', 'Output', 'Predictor',
            'Residual',
            'LS lin. approx', 'Arb. lin. approx.'),
           'upper right', prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none')
add_logo(fig, axis=ax1, location=5, size_frac_x=1/12)

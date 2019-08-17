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

# # s_normal_mean_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_normal_mean_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-3-ex-anal-resid-ii).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm

from arpym.statistics import simulate_normal
from arpym.tools import add_logo, pca_cov
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_normal_mean_regression-parameters)

mu_xz = np.array([0, 0])  # joint expectation
# dispersion parameters
rho_xz = -0.5
sigma_x = 1
sigma_z = 1
j_ = 20000  # number of simulations
beta_arb = 2  # arbitrary loading

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_normal_mean_regression-implementation-step01): Generate samples

# +
sigma2_xz = np.array([[sigma_x**2, rho_xz*sigma_x*sigma_z],
                     [rho_xz*sigma_x*sigma_z, sigma_z**2]])  # joint covariance

x, z = simulate_normal(mu_xz, sigma2_xz, j_).T
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_normal_mean_regression-implementation-step02): Compute the regression coefficients

# +
beta = sigma_x*rho_xz / sigma_z
alpha = mu_xz[0] - beta * mu_xz[1]
alpha_arb = mu_xz[0] - beta_arb * mu_xz[1]

x_reg = alpha+beta*z
u = x - x_reg
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_normal_mean_regression-implementation-step03): Expectation and covariance of (Xreg, U)

# +
a = np.array([alpha, -alpha]).reshape(-1, 1)
b = np.array([[0, beta], [1, -beta]])

mu_xreg_u = a + b @ np.reshape(mu_xz, (-1, 1))
sigma2_xreg_u = b @ sigma2_xz @ b.T
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_normal_mean_regression-implementation-step04): Expectation-covariance ellipsoid computations

# +
# covariance matrix decomposition
e, lambda2 = pca_cov(sigma2_xreg_u)
diag_lambda = np.diagflat(np.sqrt(lambda2))

# expectation-covariance ellipsoid computations
theta = np.linspace(0, 2*np.pi, 200)  # angle
y = [np.cos(theta), np.sin(theta)]  # circle parametrization
axes_points = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
ellipse = mu_xreg_u + e @ diag_lambda @ y
axes_points_transformed = mu_xreg_u + e @ diag_lambda @ axes_points
# -

# ## Plots

# +
# Compute the rectangles vertices and edges


# helper functions
def regression_opt(var):
    return alpha + beta*var


def regression_arb(var):
    return alpha_arb + beta_arb*var


# select 2 simulations closest to the points below to display squared distance from
multip = 0.5
anchor_point_1 = mu_xz + [2*multip*rho_xz, -multip*rho_xz]
anchor_point_2 = mu_xz + [-2*multip*rho_xz, multip*rho_xz]
index1 = np.argmin(np.sum(([x, z]-np.reshape(anchor_point_1, (2, 1)))**2,
                          axis=0))
index2 = np.argmin(np.sum(([x, z]-np.reshape(anchor_point_2, (2, 1)))**2,
                          axis=0))
chi_z_val_select1 = regression_opt(z[index1])
square1_edge = chi_z_val_select1-x[index1]
chi_z_val_select2 = regression_opt(z[index2])
square2_edge = chi_z_val_select2-x[index2]
arb_fun_select1 = regression_arb(z[index1])
square3_edge = arb_fun_select1-x[index1]
arb_fun_select2 = regression_arb(z[index2])
square4_edge = arb_fun_select2-x[index2]


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


fig = plt.figure(dpi=72)
upper_limit = max(np.percentile(x, 99.9), np.percentile(z, 99.9))
lower_limit = min(np.percentile(x, 0.1), np.percentile(z, 0.1))
n_grid = 100
x_grid = np.linspace(lower_limit-2*(mu_xz[1] - lower_limit),
                     mu_xz[1] + 2*(upper_limit - mu_xz[1]), n_grid)

ax1 = plt.subplot2grid((16, 10), (0, 1), colspan=6, rowspan=12)
ax1.set_aspect('equal')
ax1.set_xlim([lower_limit, upper_limit])
ax1.set_ylim([lower_limit, upper_limit])
ax1.set_xticks(isinteger(ax1.get_xticks()))
ax1.set_yticks(isinteger(ax1.get_yticks()))
ax1.tick_params(axis='both', which='major', pad=-20)
ax1.set_facecolor('none')
ax1.grid(False)
ax1.set_xlabel('$Z$', position=(1, 0), fontdict={'size': 16}, labelpad=-20)
ax1.set_ylabel('$X$', position=(0, 1), fontdict={'size': 16}, labelpad=-20)
xlimm1 = ax1.get_xlim()
ax1.scatter(z[:j_plot], x[:j_plot], s=markersize, c=[light_grey])
l5, = ax1.plot(x_grid, regression_opt(x_grid), c=orange, lw=2)
l6, = ax1.plot(x_grid, regression_arb(x_grid), c='b', lw=2)
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
# text box x_reg
x_limits = ax1.get_xlim()
y_limits = ax1.get_ylim()
ax1.annotate(r'$x^{\mathit{Reg}}=\alpha+\beta z$', fontsize=20, color=orange,
             horizontalalignment='left',
             xy=(x_limits[0]+0.2*(x_limits[1]-x_limits[0]),
                 regression_opt(x_limits[0]+0.2*(x_limits[1]-x_limits[0]))),
             xytext=(x_limits[0]+0.2*(x_limits[1]-x_limits[0]),
                     y_limits[0]+0.2*(y_limits[1]-y_limits[0])),
             arrowprops=dict(facecolor=orange, ec='none', shrink=0.05))
plt.box(False)

ax2 = plt.subplot2grid((16, 10), (0, 0), colspan=1, rowspan=12, sharey=ax1)
ax2.set_ylim([lower_limit, upper_limit])
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

ax3 = plt.subplot2grid((16, 10), (12, 1), colspan=6, rowspan=2, sharex=ax1)
ax3.set_xlim([lower_limit, upper_limit])
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

ax4 = plt.subplot2grid((16, 20), (0, 15), colspan=5, rowspan=5)
ax4.set_facecolor('none')
ax4.grid(False)
ax4.set_xlabel(r'$X^{\mathit{Reg}}$', position=(0.9, 0), fontdict={'size': 16},
               labelpad=-30)
ax4.set_ylabel(r'$X$', position=(0, 1), fontdict={'size': 16}, labelpad=-30)
ax4.tick_params(axis='both', which='major', pad=-30)
ax4.set_aspect('equal')
ax4.scatter(x_reg[:j_plot], x[:j_plot], s=markersize, c=[light_grey])
ax4.set_xlim([lower_limit, upper_limit])
ax4.set_ylim([lower_limit, upper_limit])
ax4.set_xticks(isinteger(ax4.get_xticks()))
ax4.set_yticks(isinteger(ax4.get_yticks()))
plt.box(False)

ax5 = plt.subplot2grid((16, 20), (0, 14), colspan=1, rowspan=5, sharey=ax4)
ax5.set_xticks([])
ax5.set_facecolor('none')
ax5.grid(False)
ax5.yaxis.set_ticks_position('none')
plt.setp(ax5.get_yticklabels(), visible=False)
ax5.set_ylim([lower_limit, upper_limit])
ax5.invert_xaxis()
ax5.hist(x, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
l3, = ax5.plot(norm.pdf(x_grid, mu_xreg_u[0],
                        np.sqrt(abs(sigma2_xreg_u[0, 0]))), x_grid,
               c=light_green_1, lw=2)
plt.box(False)

ax6 = plt.subplot2grid((16, 20), (5, 15), colspan=5, rowspan=1, sharex=ax4)
ax6.set_yticks([])
ax6.set_facecolor('none')
ax6.grid(False)
ax6.xaxis.set_ticks_position('none')
plt.setp(ax6.get_xticklabels(), visible=False)
ax6.set_xlim([lower_limit, upper_limit])
ax6.invert_yaxis()
ax6.hist(x_reg, bins='auto', density=True,
         facecolor=light_green_1, ec=light_green_1)

plt.box(False)

ax7 = plt.subplot2grid((16, 20), (6, 15), colspan=5, rowspan=5)
ax7.set_facecolor('none')
ax7.grid(False)
ax7.set_xlabel(r'$X^{\mathit{Reg}}$', position=(0.9, 0), fontdict={'size': 16},
               labelpad=-30)
ax7.set_ylabel(r'$U$', position=(0, 1), fontdict={'size': 16}, labelpad=-30)
ax7.tick_params(axis='both', which='major', pad=-20)
ax7.set_aspect('equal')
ax7.scatter(x_reg[:j_plot], u[:j_plot], s=markersize, c=[light_grey])
ax7.plot(ellipse[0], ellipse[1],
         axes_points_transformed[0, 0:2], axes_points_transformed[1, 0:2],
         axes_points_transformed[0, 2:], axes_points_transformed[1, 2:],
         c='k', lw=1)
ax7.set_xlim([lower_limit, upper_limit])
ax7.set_ylim([lower_limit, upper_limit])
ax7.set_xticks(isinteger(ax7.get_xticks()))
ax7.set_yticks(isinteger(ax7.get_yticks()))
plt.box(False)

ax8 = plt.subplot2grid((16, 20), (6, 14), colspan=1, rowspan=5, sharey=ax7)
ax8.set_xticks([])
ax8.set_facecolor('none')
ax8.grid(False)
ax8.yaxis.set_ticks_position('none')
plt.setp(ax8.get_yticklabels(), visible=False)
ax8.set_ylim([lower_limit, upper_limit])
ax8.invert_xaxis()
ax8.hist(u, bins='auto', density=True, facecolor=colf, ec=colf,
         orientation='horizontal')
plt.box(False)

ax9 = plt.subplot2grid((16, 20), (11, 15), colspan=5, rowspan=1, sharex=ax7)
ax9.set_yticks([])
ax9.set_facecolor('none')
ax9.grid(False)
ax9.xaxis.set_ticks_position('none')
plt.setp(ax9.get_xticklabels(), visible=False)
ax9.set_xlim([lower_limit, upper_limit])
ax9.invert_yaxis()
plt.box(False)

ax9.hist(x_reg, bins='auto', density=True,
         facecolor=light_green_1, ec=light_green_1)

l1 = Rectangle((0, 0), 1, 1, color=light_green_2, ec='none')
l2 = Rectangle((0, 0), 1, 1, color=teal, ec='none')
l4 = Rectangle((0, 0), 1, 1, color=colf, ec='none')
fig.legend((l1, l2, l3, l4, l5, l6),
           ('Factor', 'Target', 'Reg. recovered', 'Residual',
            'LS lin. approx', 'Arb. lin. approx.'),
           'upper right', prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none')
add_logo(fig, axis=ax1, location=5, size_frac_x=1/12)

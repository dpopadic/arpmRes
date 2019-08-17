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

# # s_logn_mean_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_logn_mean_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_regression).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arpym.tools.logo import add_logo
from arpym.statistics import simulate_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_regression-parameters)

# +
mu_xz = [0, 0]  # location parameter
# dispersion parameters
rho_xz = -0.5
sig_x = 0.92
sig_z = 0.85
j_ = 10**4  # number of simulations


def chi_arb(var):
    return 1/var
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_regression-implementation-step01): Generate samples

# +
sig2_xz = np.array([[sig_x**2, rho_xz*sig_x*sig_z],
                    [rho_xz*sig_x*sig_z, sig_z**2]])
# jointly lognormal samples
x, z = np.exp(simulate_normal(mu_xz, sig2_xz, j_).T)

no_points_grid = 500
x_grid = np.linspace(10**-6, 2*max(np.percentile(x, 95), np.percentile(z, 95)),
                     no_points_grid)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_regression-implementation-step02): Compute prediction, residuals and E{X|z}

# +

def chi(var):
    return np.exp(mu_xz[0]+rho_xz*sig_x/sig_z*(np.log(var)-mu_xz[1]) +
                  0.5*(1-rho_xz**2)*sig_x)


xhat = chi(z)
u = x-xhat
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_regression-implementation-step03): Expectation and covariance of (Xhat, U)

# +
# expectation and covariance of (lnXhat, lnX)
a = mu_xz[0]-mu_xz[1]*rho_xz*sig_x/sig_z+0.5*(1-rho_xz**2)*sig_x**2
b = rho_xz*sig_x/sig_z
mu_logxhat_logx = np.array([a+b*mu_xz[1], mu_xz[0]])
sig2_logxhat_logx = [[b**2*sig_z**2, b*rho_xz*sig_x*sig_z],
                     [b*rho_xz*sig_x*sig_z, sig_x**2]]
sig2_logxhat_logx = np.array(sig2_logxhat_logx)

# expectation and covariance of (Xhat,X)
mu_xhat_x = np.exp(mu_logxhat_logx+0.5*np.diag(sig2_logxhat_logx))
sig2_xhat_x = np.diag(mu_xhat_x)@\
                    (np.exp(sig2_logxhat_logx) - np.ones((2, 2)))@\
                    np.diag(mu_xhat_x)

# expectation and covariance of (Xhat,U)
d = np.array([[1, 0], [-1, 1]])  # (Xhat, U)=d*(Xhat, X)
mu_xhat_u = d@mu_xhat_x
sig2_xhat_u = d@sig2_xhat_x@d.T
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_logn_mean_regression-implementation-step04): expectation-covariance ellipsoid computations

# +
# matrix decomposition
lambda2, e = np.linalg.eig(sig2_xhat_u)
lambda2, order = np.sort(lambda2), np.argsort(lambda2)
e = e[:, order]
diag_lambda = np.diagflat(np.sqrt(lambda2))

# expectation-covariance ellipsoid computations
theta = np.linspace(0, 2*np.pi, no_points_grid)  # angle
y = [np.cos(theta), np.sin(theta)]  # circle parametrization
axes_points = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
ellipse = mu_xhat_u.reshape((2, 1)) + e@diag_lambda@y
axes_points_transformed = mu_xhat_u.reshape((2, 1)) + e@diag_lambda@axes_points
# -

# ## Plots:

# +
# Compute the rectangles vertices and edges
# select 2 simulations closest to the points below to display squared distance from
anchor_point_1 = [0.5, 0.3]
anchor_point_2 = [1.5, 6]
index1 = np.argmin(np.sum(([x, z]-np.reshape(anchor_point_1, (2, 1)))**2,
                          axis=0))
index2 = np.argmin(np.sum(([x, z]-np.reshape(anchor_point_2, (2, 1)))**2,
                          axis=0))
chi_z_val_select1 = chi(z[index1])
square1_edge = chi_z_val_select1-x[index1]
chi_z_val_select2 = chi(z[index2])
square2_edge = chi_z_val_select2-x[index2]
arb_fun_select1 = chi_arb(z[index1])
square3_edge = arb_fun_select1-x[index1]
arb_fun_select2 = chi_arb(z[index2])
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
markersize = 6
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
ax1.set_xlabel('$Z$', position=(1, 0), fontdict={'size': 16}, labelpad=-40)
ax1.set_ylabel('$X$', position=(0, 1), fontdict={'size': 16}, labelpad=-40)
xlimm1 = ax1.get_xlim()
ax1.scatter(z[:j_plot], x[:j_plot], s=markersize, c=[light_grey])
l5, = ax1.plot(x_grid, chi(x_grid), c=orange, lw=2)
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
ax1.text((1-1/4)*xlimm1[1], 1.5*chi(xlimm1[1]), '$\hat{x}=\chi(z)$',
         fontdict={'color': orange, 'size': 20})
plt.box(False)

ax2 = plt.subplot2grid((10, 16), (0, 1), colspan=2, rowspan=6)
ax2.set_ylim([0, upper_limit])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_facecolor('none')
ax2.grid(False)
ax2.invert_xaxis()
ax2.hist(x, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
plt.box(False)

ax3 = plt.subplot2grid((10, 16), (6, 3), colspan=12, rowspan=1)
ax3.set_xlim([0, 1.53*upper_limit])
ax3.set_xticks(ax3.get_xticks()[1:])
ax3.set_xticks([])
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
ax5.hist(x, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
ax5.plot(logn_pdf(x_grid,
                  mu_xz[0]+0.5*(1-rho_xz**2)*sig_x**2 -
                  mu_xz[1]*rho_xz*sig_x/sig_z,
                  abs(rho_xz*sig_x)), x_grid, c=light_green_1, lw=2)
plt.box(False)

ax6 = plt.subplot2grid((20, 16), (19, 1), colspan=7, rowspan=1, sharex=ax4)
ax6.set_yticks([])
ax6.set_facecolor('none')
ax6.grid(False)
ax6.xaxis.set_ticks_position('none')
plt.setp(ax6.get_xticklabels(), visible=False)
ax6.set_xlim([0, 2*upper_limit])
ax6.invert_yaxis()
ax6.plot(x_grid, logn_pdf(x_grid,
                          mu_xz[0]+0.5*(1-rho_xz**2)*sig_x**2 -
                          mu_xz[1]*rho_xz*sig_x/sig_z,
                          abs(rho_xz*sig_x)), c=light_green_1, lw=2)
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
l3, = ax9.plot(x_grid, logn_pdf(x_grid,
                                mu_xz[0]+0.5*(1-rho_xz**2)*sig_x**2 -
                                mu_xz[1]*rho_xz*sig_x/sig_z,
                                abs(rho_xz*sig_x)), c=light_green_1, lw=2)

l1 = Rectangle((0, 0), 1, 1, color=light_green_2, ec='none')
l2 = Rectangle((0, 0), 1, 1, color=teal, ec='none')
l4 = Rectangle((0, 0), 1, 1, color=colf, ec='none')
fig.legend((l1, l2, l3, l4, l5, l6),
           ('Input', 'Output', 'Predictor',
            'Residual',
            'Cond. exp.', 'Arb. func.'),
           'upper right', prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none')
add_logo(fig, axis=ax1, location=5, size_frac_x=1/12)

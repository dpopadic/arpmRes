#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_supervised_classification [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_supervised_classification&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification).

# +
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm

from arpym.statistics import saddle_point_quadn
from arpym.tools import plot_ellipse, add_logo

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification-parameters)

p = 0.4  # unconditional output probability
q = 0.7  # false positive error weight
mu_z_0 = np.array([0, -1])  # location when X=0
sig2_z_0 = np.array([[49, -12], [-12, 36]])  # dispersion when X=0
mu_z_1 = np.array([1, 5])  # location when X=1
sig2_z_1 = np.array([[36, 11], [11, 49]])  # dispersion when X=1
a = 0  # parameter for arbitrary linear score
b = np.array([2, 0])  # parameter for arbitrary linear score
j_ = 10 ** 3  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification-implementation-step01): Conditional optimal score parameters, optimal scoring function and optimal predictor

# +
# parameters of the optimal scoring function
alpha = np.log(p / (1 - p)) - \
        0.5 * (np.log(np.linalg.det(np.linalg.solve(sig2_z_0, sig2_z_1))) +
               mu_z_1 @ np.linalg.solve(sig2_z_1, mu_z_1) -
               mu_z_0 @ np.linalg.solve(sig2_z_0, mu_z_0))
beta = np.linalg.solve(sig2_z_1, mu_z_1) - np.linalg.solve(sig2_z_0, mu_z_0)
gamma = -0.5 * (np.linalg.solve(sig2_z_1, np.identity(sig2_z_1.shape[0])) -
                np.linalg.solve(sig2_z_0, np.identity(sig2_z_0.shape[0])))

# the optimal scoring function
s_star = lambda z: alpha + beta @ z + z.T @ gamma @ z

# the optimal point predictor
lnq = np.log(q)
chi = lambda z: s_star(z) > lnq
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification-implementation-step02): Distribution of conditional optimal score, false positve rate and true positive rate of optimal scor, optimal ROC curve

# +
# cdf and pdf of conditional optimal score
cdf_s_star_given_0 = lambda s_: \
    saddle_point_quadn(s_, alpha, beta.T, gamma, mu_z_0, sig2_z_0)[0]
pdf_s_star_given_0 = lambda s_: \
    saddle_point_quadn(s_, alpha, beta.T, gamma, mu_z_0, sig2_z_0)[1]
cdf_s_star_given_1 = lambda s_: \
    saddle_point_quadn(s_, alpha, beta.T, gamma, mu_z_1, sig2_z_1)[0]
pdf_s_star_given_1 = lambda s_: \
    saddle_point_quadn(s_, alpha, beta.T, gamma, mu_z_1, sig2_z_1)[1]

# false positive rate and true positive rate of optimal score
fpr = lambda s_: 1 - cdf_s_star_given_0(s_)
tpr = lambda s_: 1 - cdf_s_star_given_1(s_)

# optimal false positive rate and true positive rate
fpr_star = fpr(lnq)
tpr_star = tpr(lnq)

# optimal ROC curve
roc = lambda s_: [fpr(s_), tpr(s_)]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification-implementation-step03): Distribution of conditional linear score, false positve rate and true positive rate of linear scor, ROC curve of linear score

# +
# cdf of conditional linear score
cdf_s_given_0 = lambda s_: \
    norm.cdf(s_, a + b @ mu_z_0, np.sqrt(b @ sig2_z_0 @ b.T))
cdf_s_given_1 = lambda s_: \
    norm.cdf(s_, a + b @ mu_z_1, np.sqrt(b @ sig2_z_1 @ b.T))

# false positive rate and true positive rate of linear score
fpr_lin = lambda s_: 1 - cdf_s_given_0(s_)
tpr_lin = lambda s_: 1 - cdf_s_given_1(s_)

# ROC curve of linear score
roc_lin = lambda s_: [fpr_lin(s_), tpr_lin(s_)]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification-implementation-step04): Generate simulations

# +
x = np.random.binomial(1, p, j_)
z_given_0 = np.random.multivariate_normal(mu_z_0, sig2_z_0, j_)
z_given_1 = np.random.multivariate_normal(mu_z_1, sig2_z_1, j_)

z = np.empty_like(z_given_0)
for j in range(j_):
    z[j] = (1 - x[j]) * z_given_0[j] + x[j] * z_given_1[j]

x_bar = np.empty_like(x)
for j in range(j_):
    x_bar[j] = chi(z[j])
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_supervised_classification-implementation-step05): Generate mean/covariance ellipses

ellipse_0 = plot_ellipse(mu_z_0, sig2_z_0, r=2, display_ellipse=False)
ellipse_1 = plot_ellipse(mu_z_1, sig2_z_1, r=2, display_ellipse=False)

# ## Plots

# +
green = [0, 0.5, 0]
teal = [0.2344, 0.5820, 0.5664]
light_teal = [0.2773, 0.7031, 0.6836]
light_green_1 = [0.8398, 0.9141, 0.8125]
light_green_2 = [0.6, 0.8008, 0.5039]
grey = [0.5, 0.5, 0.5]
colf = [0, 0.5412, 0.9020]
darkgrey = [0.3, 0.3, 0.3]
color_multiplier = 0.6
x_colors = np.empty_like(x)
x_colors[(x == 1) & (x_bar == 1)] = 1
x_colors[(x == 1) & (x_bar == 0)] = 2
x_colors[(x == 0) & (x_bar == 0)] = 3
x_colors[(x == 0) & (x_bar == 1)] = 4

# grid for optimal roc curve
s_min = -20
s_max = 20
s_star_grid = np.arange(s_min, s_max, 0.05)
s_star_grid_index = np.argmin(abs(lnq - s_star_grid))

# grid for linear roc curve
s_lin_grid = np.linspace(-500, 500, 1000)

plt.style.use('arpm')
fig = plt.figure()

ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=1)
divider = make_axes_locatable(ax1)

# dummy legend plots
ax1.fill_between([5, 5], [5, 5], [5, 5], facecolor=light_green_1, label='fnr')
ax1.fill_between([5, 5], [5, 5], [5, 5], facecolor=light_green_2, label='tpr')
ax1.fill_between([5, 5], [5, 5], [5, 5], facecolor=teal, label='tnr')
ax1.fill_between([5, 5], [5, 5], [5, 5], facecolor=light_teal, label='fpr')

# Optimal ROC curve and classifier, arbitrary ROC curve, Neyman-Pearson region
fpr_grid = fpr(s_star_grid)
tpr_grid = tpr(s_star_grid)

fpr_lin_grid = fpr_lin(s_lin_grid)
tpr_lin_grid = tpr_lin(s_lin_grid)

ax1.plot(fpr_grid, tpr_grid, color=green, lw=0.75, label='Best ROC')
ax1.plot(fpr_star, tpr_star,
         'ro', markersize=8, label='Optimal classifier')
ax1.plot(fpr_lin_grid, tpr_lin_grid, 'k--', linewidth=0.75, label='Arbitrary ROC')
ax1.fill(1 - fpr_grid, 1 - tpr_grid, color='grey', alpha=0.3)
ax1.fill(fpr_grid, tpr_grid, color='grey', alpha=0.3)

# non-predictive classifier
ax1.plot([0, 1], [0, 1], 'k', lw=0.75, label='Non-predictive classifier')
ax1.axis('square')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel('FPR (and TNR)', labelpad=18)
ax1.set_ylabel('TPR (and FNR)', labelpad=22)
ax1.legend(facecolor='none', edgecolor='none', loc=4, ncol=2)
ax1.set_title('ROC curve')
ax1.set_xticks([])
ax1.set_yticks([])

# TPR
ax2 = divider.append_axes('left', size='3%', pad=0)
ax2.set_ylim([0, 1])
ax2.fill_between([0, 1], [1, 1], 0, facecolor=light_green_1)
ax2.fill_between([0, 1], [tpr_star, tpr_star], 0, facecolor=light_green_2)
ax2.set_xticks([])
ax2.set_yticks(np.arange(0, 1, 0.2))
ax2.tick_params(axis='both', which='major', pad=0)

# FPR
ax3 = divider.append_axes('bottom', size='3%', pad=0)
ax3.set_xlim([0, 1])
ax3.fill_between([0, 1], [1, 1], 0, facecolor=teal)
ax3.fill_between([0, fpr_star], [1, 1], 0,
                 facecolor=light_teal)
ax3.set_xticks(np.arange(0, 1, 0.2))
ax3.set_yticks([])
ax3.tick_params(axis='both', which='major', pad=0)

ax4 = plt.subplot2grid((6, 2), (4, 0), rowspan=1, colspan=1)

# Optimal score distribution S|0
pdf_s_star_given_0_grid = pdf_s_star_given_0(s_star_grid)
ax4.fill_between(s_star_grid[s_star_grid_index:],
                 pdf_s_star_given_0_grid[s_star_grid_index:],
                 0, facecolor=light_teal)
ax4.fill_between(s_star_grid[:s_star_grid_index],
                 pdf_s_star_given_0_grid[:s_star_grid_index:],
                 0, facecolor=teal)
ax4.plot(s_star_grid, pdf_s_star_given_0_grid, c=[x * color_multiplier for x in light_green_2], lw=2)
ax4.annotate(r'Optimal score $S^*|0$', (0.06, 0.31),
             xycoords='figure fraction', size=14.5)
ylimm = ax4.get_ylim()
ax4.plot([lnq, lnq], ylimm, c=colf, lw=1.5)
ax4.text(lnq, 0.8 * ylimm[1], r'$\ln q$')
ax4.set_xticks([])
ax4.set_xlim([s_min, s_max])
ax4.grid(False)

ax5 = plt.subplot2grid((6, 2), (5, 0), rowspan=1, colspan=1)

# Optimal score distribution S|1
pdf_s_star_given_1_grid = pdf_s_star_given_1(s_star_grid)
ax5.fill_between(s_star_grid[s_star_grid_index:],
                 pdf_s_star_given_1_grid[s_star_grid_index:],
                 0, facecolor=light_green_2)
ax5.fill_between(s_star_grid[:s_star_grid_index],
                 pdf_s_star_given_1_grid[:s_star_grid_index:],
                 0, facecolor=light_green_1)
ax5.plot(s_star_grid, pdf_s_star_given_1_grid, c=[x * color_multiplier for x in light_green_2], lw=2)
ax5.plot([lnq, lnq], ylimm, c=colf, lw=1.5)
ax5.set_xlim([s_min, s_max])
ax5.annotate(r'Optimal score $S^*|1$', (0.06, 0.16),
             xycoords='figure fraction', size=14.5)
ax5.grid(False)

ax6 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1, projection='3d')

# Optimal score
zlim = [-40, 40]
z_grid = np.arange(zlim[0], zlim[1], 0.5)
i_ = z_grid.size
s_star_meshgrid = np.empty([i_, i_])
for i in range(i_):
    for j in range(i_):
        s_star_meshgrid[i, j] = s_star(np.array([z_grid[i], z_grid[j]]))

z_region_meshgrid = s_star_meshgrid > lnq
z_column, z_row = np.meshgrid(z_grid, z_grid)

ax6.plot3D(z_column[z_region_meshgrid],
           z_row[z_region_meshgrid],
           lnq * np.ones_like(s_star_meshgrid[z_region_meshgrid]),
           's', ms=4, c=colf, alpha=0.15)
ax6.contour(z_column, z_row, s_star_meshgrid, levels=np.arange(-100, 100, 5),
            colors=[[x * color_multiplier for x in light_green_2]],
            linewidths=2, linestyles=['solid'])
ax6.set_xlim(zlim)
ax6.set_ylim(zlim)
ax6.set_zlim([-100, 100])
ax6.set_xlabel(r'$z_{1}$')
ax6.set_ylabel(r'$z_2$')
ax6.set_zlabel(r'$s$')
ax6.set_title(r'Optimal score $s^*(z)$')

ax7 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1, projection='3d')

# Optimal predictor
ax7.plot3D(z_column[z_region_meshgrid],
           z_row[z_region_meshgrid],
           np.ones_like(z_column[z_region_meshgrid]),
           's', ms=4, c=[0.8, 0.8, 0.8], alpha=0.3)
ax7.plot3D(z_column[~z_region_meshgrid],
           z_row[~z_region_meshgrid],
           np.zeros_like(z_column[~z_region_meshgrid]),
           's', ms=4, c=[0.8, 0.8, 0.8], alpha=0.3)
ax7.plot(z[:, 0][x_colors == 1], z[:, 1][x_colors == 1], x[x_colors == 1],
         'o', ms=3, c=light_green_2)
ax7.plot(z[:, 0][x_colors == 2], z[:, 1][x_colors == 2], x[x_colors == 2],
         'o', ms=3, c=light_green_1)
ax7.plot(z[:, 0][x_colors == 3], z[:, 1][x_colors == 3], x[x_colors == 3],
         'o', ms=3, c=teal)
ax7.plot(z[:, 0][x_colors == 4], z[:, 1][x_colors == 4], x[x_colors == 4],
         'o', ms=3, c=light_teal)

# mean/covariance ellipses
ax7.plot(ellipse_0[:, 0], ellipse_0[:, 1], np.zeros_like(ellipse_0[:, 0]), c='k', lw=.5)
ax7.plot(ellipse_1[:, 0], ellipse_1[:, 1], np.ones_like(ellipse_1[:, 0]), c='k', lw=0.5)
ax7.plot([mu_z_0[0]], [mu_z_0[1]], [0], 'o', ms=3, c='k')
ax7.plot([mu_z_1[0]], [mu_z_1[1]], [1], 'o', ms=3, c='k')
ax7.set_xlim(zlim)
ax7.set_ylim(zlim)
ax7.set_zlim([0, 1.2])
ax7.set_zticks([0, 1])
ax7.set_xlabel(r'$z_{1}$')
ax7.set_ylabel(r'$z_2$')
ax7.set_zlabel(r'$x$')
ax7.set_title(r'Optimal predictor $\hat{x}=1_{s^*(z)>\bar{s}^*}$')

add_logo(fig, size_frac_x=1 / 9)

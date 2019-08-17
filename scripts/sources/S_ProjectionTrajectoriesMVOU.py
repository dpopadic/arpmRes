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

# # S_ProjectionTrajectoriesMVOU [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ProjectionTrajectoriesMVOU&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-mvou-covariance-evolution).

# ## Prepare the environment

# +
import os.path as path
import sys,os

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, imag, array, ones, zeros, cos, sin, pi, where, linspace, diag, \
    sqrt, tile, r_, real, diagflat
from numpy.linalg import eig, solve

from scipy.linalg import block_diag

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, legend, xlim, ylim, subplots, xlabel, title, xticks
from matplotlib import gridspec
from matplotlib.pyplot import ylabel
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')

from pcacov import pcacov
from ARPM_utils import save_plot
from ProjMomentsVAR1MVOU import ProjMomentsVAR1MVOU

# See also A. Meucci (2009)
# "Review of Statistical Arbitrage, Cointegration, and Multivariate Ornstein-Uhlenbeck"
# available at ssrn.com

# input parameters

# horizon
t_end = 0.3  # 0.3 y
dt = 1 / 400
horiz_u = arange(0,t_end+dt,dt)
u_ = len(horiz_u)

k_ = 1  # number of real eigenvalues
j_ = 1  # pairs of complex eigenvalues
mu = zeros((k_ + 2*j_, 1))  # drift vector
theta = array([[-10 ** -5, -120, -10], [- 120, 10, 210], [- 10, -210, 10]])  # transition matrix
sigma = array([[0.50, 0.7071, 0.50], [0.7071, 2.0, 1.0], [0.50, 1.0, 1.0]])  # scatter generator
sigma2 = sigma@sigma.T
x_0 = ones((k_ + 2*j_, 1))  # initial value
# -

# ##  Compute the real block-diagonal matrix gamma, the matrix s
# ## and the initial value z_0 in the diagonalized coordinates.

lam, beta = eig(theta)  # eigenvectors and eigenvalues of theta
alpha = real(beta) - imag(beta)  # real matrix of eigenvectors
gamma_ja = real(lam[1])
gamma_jb = imag(lam[1])
gamma = block_diag(lam[0], array([[gamma_ja, gamma_jb], [-gamma_jb, gamma_ja]]))  # real diagonal-block matrix
gamma = real(gamma)
z_0 = solve(alpha,x_0 - solve(theta,mu))
s = solve(alpha,sigma)
s2 = s@s.T

# ## Project the conditional first and second moments of the MVOU process at future horizons using function ProjMomentsVAR1MVOU in the original coordinates
# ### and compute the location-dispersion ellipsoid and the corresponding principal components.

# +
th = linspace(0,2*pi,21)
i_ = len(th)
ph = linspace(0,2*pi,21)
p_ = len(ph)
ell_x1 = zeros((i_, p_, u_))
ell_x1a = zeros((i_, p_, u_))
ell_x1b = zeros((i_, p_, u_))
princdir_x = zeros((k_ + 2*j_, k_ + 2*j_, u_))  # principal direction

x_mu_u, x_sigma2_u, x_drift_u = ProjMomentsVAR1MVOU(x_0, horiz_u, mu, theta, sigma2)

for u in range(1,u_):
    [eigvec_x, eigval_x] = pcacov(x_sigma2_u[:,:, u])
    for i in range(i_):
        for p in range(p_):
            y_x =r_[sin(th[i])*cos(ph[p]), sin(th[i])*sin(ph[p]), cos(th[i])]
            # compute the location-dispersion ellipsoid
            ellipsoid_x = x_drift_u[:,u] + eigvec_x@diag(sqrt(eigval_x))@y_x
            ell_x1[i, p, u] = ellipsoid_x[0]
            ell_x1a[i, p, u] = ellipsoid_x[1]
            ell_x1b[i, p, u] = ellipsoid_x[2]
            # compute the principal directions of the ellipsoid
            princdir_x[:,:, u] = tile(x_drift_u[:, [u]], (1, k_ + 2*j_)) + eigvec_x@sqrt(diagflat(eigval_x))
# -

# ## Projects the conditional first and second moments of the MVOU process at future horizons using function ProjMomentsVAR1MVOU in the diagonalized coordinates
# ### and compute the location-dispersion ellipsoid and the corresponding principal components.

# +
ell_z1 = zeros((i_, p_, u_))
ell_z1a = zeros((i_, p_, u_))
ell_z1b = zeros((i_, p_, u_))
princdir_z = zeros((k_ + 2*j_, k_ + 2*j_, u_))
z_mu_u, z_s2_u, z_drift_u = ProjMomentsVAR1MVOU(z_0, horiz_u, mu, gamma, s2)

for u in range(1,u_):
    # compute the ellipsoid
    [eigvec_z, eigval_z] = pcacov(z_s2_u[:,:, u])

    for i in range(i_):
        for p in range(p_):
            y_z =[sin(th[i])*cos(ph[p]), sin(th[i])*sin(ph[p]), cos(th[i])]
            # compute the location-dispersion ellipsoid
            ellipsoid_z = z_drift_u[:,u] + eigvec_z@diag(sqrt(eigval_z))@y_z
            ell_z1[i, p, u] = ellipsoid_z[0]
            ell_z1a[i, p, u] = ellipsoid_z[1]
            ell_z1b[i, p, u] = ellipsoid_z[2]
    # compute the principal directions of the ellipsoid
    princdir_z[:,:, u] = tile(z_drift_u[:, [u]], (1, k_ + 2*j_)) + eigvec_z@sqrt(diag(eigval_z))
# -

# ## Plot the the conditional expectation and the location-dispersion ellipsoid stemming from the covariance
# ## both in the original coordinates and in the diagonalized coordinates at the selected horizons (2 months and 4 months),
# ## along with the principal components and the current position of the conditional mean.
# ## Then plot separately each component of the conditional expectation,
# ## both in the original coordinates and in the diagonalized ones,
# ## highlighting the current position of the conditional expectation.

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
hor_sel = t_end*0.5  # 2 months
hor_sel2 = t_end  # 4 months
i1 = where(horiz_u == hor_sel)[0][0]
i2 = where(horiz_u == hor_sel2)[0][0]

for i in [i1,i2]:
    plt.figure()
    gs = gridspec.GridSpec(6, 2)
    ax11 = plt.subplot(gs[:3, 0], projection='3d')
    ax12 = plt.subplot(gs[3, 0])
    ax13 = plt.subplot(gs[4, 0])
    ax14 = plt.subplot(gs[5, 0])
    ax21 = plt.subplot(gs[:3, 1], projection='3d')
    ax22 = plt.subplot(gs[3, 1])
    ax23 = plt.subplot(gs[4, 1])
    ax24 = plt.subplot(gs[5, 1])
    # 3-d graph conditional expectation and location-dispersion ellipsoid (original coordinates)
    plt.sca(ax11)
    ax11.view_init(16, -126)
    xlim([min(x_drift_u[0]), max(x_drift_u[0])])
    ylim([min(x_drift_u[1]), max(x_drift_u[1])])
    ax11.set_zlim([min(x_drift_u[2,:]), max(x_drift_u[2,:])])
    l1 = plot(x_drift_u[0, :i], x_drift_u[1, :i], x_drift_u[2, :i])  # conditional mean
    ax11.contour(ell_x1[:,:, i], ell_x1a[:,:, i], ell_x1b[:,:, i], 15,colors=[lgrey], linewidths=0.5)  # location-dispersion ellipsoid
    # current position of the conditional mean
    l2 = ax11.plot(x_drift_u[0,[i]],x_drift_u[1,[i]],x_drift_u[2,[i]])
    # # principal directions
    l3 = ax11.plot([x_drift_u[0,[i]], princdir_x[0, 0, [i]]],[x_drift_u[1, [i]], princdir_x[1, 0, [i]]],[x_drift_u[2, i], princdir_x[2, 0, i]], c='r')
    ax11.plot([x_drift_u[0,[i]], princdir_x[0, 1, [i]]],[x_drift_u[1, [i]], princdir_x[1, 1, [i]]], [x_drift_u[2, i], princdir_x[2, 1, i]], c='r')
    ax11.plot([x_drift_u[0,[i]], princdir_x[0, 2, [i]]],[x_drift_u[1, [i]], princdir_x[1, 2, [i]]], [x_drift_u[2, i], princdir_x[2, 2, i]], c='r')
    title('Original coordinates')
    xlabel('$X_1$', labelpad=10)
    ylabel('$x_{1a}$', labelpad=10)
    ax11.set_zlabel('$x_{1b}$', labelpad=10)
    # Components fo the conditional expectation (original coordinates)
    plt.sca(ax12)
    # x_1
    xlim([min(horiz_u), max(horiz_u)])
    ylim([min(x_drift_u[0]), max(x_drift_u[0])])
    # xticks(arange(0,t_end+0.1,0.1))
    plot(horiz_u[:i], x_drift_u[0, :i])
    plot(horiz_u[[i]],x_drift_u[0,[i]],color='g',marker='.',markersize=15)
    ylabel('$x_1$',rotation=0,labelpad=10)
    # x_1a
    plt.sca(ax13)
    xlim([min(horiz_u), max(horiz_u)])
    ylim([min(x_drift_u[1]), max(x_drift_u[1])])
    # xticks(arange(0,t_end+0.1,0.1))
    plot(horiz_u[:i], x_drift_u[1, :i])
    plot(horiz_u[i], x_drift_u[1, i],color='g',marker='.',markersize=15)
    ylabel('$x_{1a}$',rotation=0,labelpad=10)
    # # x_1b
    plt.sca(ax14)
    xlim([min(horiz_u), max(horiz_u)])
    ylim([min(x_drift_u[2,:]), max(x_drift_u[2,:])])
    xticks(arange(0,t_end+0.1,0.1))
    plot(horiz_u[:i], x_drift_u[2, :i])
    plot(horiz_u[i],x_drift_u[2, i],color='g',marker='.',markersize=15)
    ylabel('$x_{1b}$',rotation=0,labelpad=10)
    xlabel('Horizon')
    # # 3-d graph conditional expectation and location-dispersion ellipsoid (diagonal coordinates)
    plt.sca(ax21)
    ax21.view_init(16, -126)
    xlim([min(z_drift_u[0]), max(z_drift_u[0])])
    ylim([min(z_drift_u[1]), max(z_drift_u[1])])
    ax21.set_zlim([min(z_drift_u[2,:]), max(z_drift_u[2,:])])
    ax21.plot(z_drift_u[0, :i], z_drift_u[1, :i], z_drift_u[2, :i])  # conditional mean
    ax21.contour(ell_z1[:,:, i], ell_z1a[:,:, i], ell_z1b[:,:, i], 15, colors=[lgrey], linewidths=0.5)  # location-dispersion ellipsoid
    # current position of the conditional mean
    ax21.plot(z_drift_u[0,[i]],z_drift_u[1,i],z_drift_u[2, i],c='g',marker='.',markersize= 15)
    # principal directions
    dir_z1 = plot([z_drift_u[0,i], princdir_z[0, 0, i]],[z_drift_u[1, i], princdir_z[1, 0, i]], [z_drift_u[2, i], princdir_z[2, 0, i]])
    dir_z1a = plot([z_drift_u[0,i], princdir_z[0, 1, i]],[z_drift_u[1, i], princdir_z[1, 1, i]], [z_drift_u[2, i], princdir_z[2, 1, i]])
    dir_z1b = plot([z_drift_u[0,i], princdir_z[0, 2, i]],[z_drift_u[1, i], princdir_z[1, 2, i]], [z_drift_u[2, i], princdir_z[2, 2, i]])
    xlabel('$Z_1$', labelpad=10)
    ylabel('$z_{1a}$', labelpad=10)
    ax21.set_zlabel('$z_{1b}$', labelpad=10)
    title('Diagonal coordinates')
    # Components of the conditional expectation (diagonal coordinates)
    # z_1
    plt.sca(ax22)
    xlim([min(horiz_u), max(horiz_u)])
    ylim([min(z_drift_u[0]), max(z_drift_u[0])])
    # xticks(arange(0,t_end+0.1,0.1))
    plot(horiz_u[:i], z_drift_u[0, :i])
    plot(horiz_u[i], z_drift_u[0,i], color='g',marker='.',markersize=15)
    ylabel('$z_1$',rotation=0,labelpad=10)
    # z_1a
    plt.sca(ax23)
    xlim([min(horiz_u), max(horiz_u)])
    ylim([min(z_drift_u[1]), max(z_drift_u[1])])
    # xticks(arange(0,t_end+0.1,0.1))
    plot(horiz_u[:i], z_drift_u[1, :i])
    plot(horiz_u[i], z_drift_u[1, i],color='g',marker='.',markersize=15)
    ylabel('$z_{1a}$',rotation=0,labelpad=10)
    # z_1b
    plt.sca(ax24)
    xlim([min(horiz_u), max(horiz_u)])
    ylim([min(z_drift_u[2,:]), max(z_drift_u[2,:])])
    plot(horiz_u[:i], z_drift_u[2, :i])
    plot(horiz_u[i],z_drift_u[2, i], color='g',marker='.',markersize=15)
    xlabel('Horizon')
    ylabel('$z_{1b}$',rotation=0,labelpad=10)
    l4 = ax24.plot(0,0,c=lgrey);
    plt.sca(ax11)
    legend(handles=[l1[0],l4[0],l3[0],l2[0]],
           labels=['Conditional expect.','Conditional covar.','Principal dir.','Current pos.'],
           bbox_to_anchor=(0., 1.01, 2.2, .122), loc='upper center',
           ncol=4, mode="expand");
    plt.tight_layout()

    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])


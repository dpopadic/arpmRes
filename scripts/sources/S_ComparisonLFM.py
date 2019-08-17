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

# # S_ComparisonLFM [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ComparisonLFM&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmtime-cor-copy-2).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, zeros, sort, argsort, squeeze, \
    linspace, diag, eye, sqrt, tile, r_
from numpy.linalg import eig

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim

plt.style.use('seaborn')

from ARPM_utils import save_plot
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid

# inputs
scale = 3
n_ = 1
k_ = 1

m_Z = 1
m_X = 1.5
m_Xemb = r_[m_X, m_Z]

s2_Z = 1.4
s_XZ = 0.7
s2_X = 1.2
s2_Xemb = array([[s2_Z, s_XZ],
             [s_XZ, s2_X]])

xemb = array([[5],[-0.5]])
# -

# ## Principal-component LFM

# +
s_vec = sqrt(diag(eye(n_ + k_)))

# compute spectral decomposition of correlation matrix
c2_Xemb = np.diagflat(1 / s_vec)@s2_Xemb@np.diagflat(1 / s_vec)

Diaglambda2, e = eig(c2_Xemb)
lambda2 = Diaglambda2
lambda2, i = sort(lambda2)[::-1], argsort(lambda2)[::-1]  # sorting
e = e[:, i]
# -

# ## compute optimal loadings and factor

# +
beta_PC = np.diagflat(s_vec)@e[:, :k_]
z_PC = e[:, :k_].T@np.diagflat(1 / s_vec)@xemb

# compute optimal coefficient a
m_Z_PC = e[:, :k_].T@np.diagflat(1 / s_vec)@m_Xemb
alpha_PC = m_Xemb - beta_PC@m_Z_PC

# compute recovered target variable
x_tilde_PC = alpha_PC.reshape(-1,1) + beta_PC@z_PC

# compute projection line and eigenvectors
step = 0.01
u_1 = arange(-2.7*scale,2.7*scale+3.6*scale / 50,3.6*scale / 50)
u_2 = arange(-1.0*scale + step, 1.0*scale, step)
r1_ = len(u_1)
r2_ = len(u_2)
pc_line = zeros((2, r1_))
princ_dir1 = zeros((2, int((r2_ + 1) / 2)))
princ_dir2 = zeros((2, int((r2_ + 1) / 2)))
for r1 in range(r1_):  # line
    pc_line[:,r1] = alpha_PC + e[:, 0]*u_1[r1]

for r2 in range(int((r2_ + 1) / 2)):  # eigenvectors
    princ_dir1[:, r2] = m_Xemb + e[:, 0]*sqrt(lambda2[0])*u_2[r2]
    princ_dir2[:, r2] = m_Xemb + e[:, 1]*sqrt(lambda2[1])*u_2[r2]
# -

# ## Regression LFM

# +
# compute optimal loadings
beta_Reg = s_XZ/s2_Z
# compute optimal coefficient a
alpha_Reg = m_X - beta_Reg*m_Z

# compute recovered target variable
x_b_Reg = alpha_Reg + beta_Reg*xemb[0]
x_tilde_Reg = r_[xemb[0], x_b_Reg]

# compute projection line
reg_line = zeros((2, 51))
reg_line[0] = linspace(m_Z - 1.5*scale*sqrt(s2_Z), m_Z + 1.5*scale*sqrt(s2_Z),51)
l_ = len(squeeze(reg_line[0]))
reg_line[1] = tile(alpha_Reg, (1, l_)) + beta_Reg*reg_line[0]
# -

# ## Create figure

# +
figure(figsize=(10,10))

# Reg line
h1 = plot(reg_line[0], reg_line[1], 'b')
# PC line
h2 = plot(pc_line[0], pc_line[1], 'g')
# eigenvectors
h3 = plot(princ_dir1[0], princ_dir1[1], 'm')
plot(princ_dir2[0], princ_dir2[1], 'm')

e1_ell = e[:, [0]]*sqrt(lambda2[0])*u_2[0]
e2_ell = e[:, [1]]*sqrt(lambda2[1])*u_2[0]
mat_ell = r_['-1',e1_ell, e2_ell]
mat_ell = mat_ell@mat_ell.T
PlotTwoDimEllipsoid(array([[princ_dir2[0, 299], princ_dir2[1, 299]]]).T, mat_ell, 1, color='g', linewidth=1)

legend(['Regression line','PC line','Principal axes'])

# data
dx = 0.2
plot(xemb[0], xemb[1], marker='.',markersize=10, color='k')
# Reg projection
plot(x_tilde_Reg[0], x_tilde_Reg[1], marker='.',markersize=10, color='k')
# PC projection
plot(x_tilde_PC[0], x_tilde_PC[1], marker='.',markersize=10, color='k')
plt.text(xemb[0] + dx, xemb[1] + dx,'$x^{\mathit{emb}}$' )
plt.text(x_tilde_Reg[0] - 1.5*dx, x_tilde_Reg[1] + 1.5*dx,r'$\tilde{x}^{\mathit{Reg}}$')
plt.text(x_tilde_PC[0] - 1.5*dx, x_tilde_PC[1] + 1.5*dx,r'$\tilde{x}^{\mathit{PC}}$')

xlim([m_Z - 3.75*s2_Z, m_Z + 3.75*s2_Z])
ylim([m_X - 4*s2_X, m_X + 4*s2_X]);
plt.axis('equal');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])


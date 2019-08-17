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

# # s_linclass_fda [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_linclass_fda&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_fda).

# +
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import auc, confusion_matrix, roc_curve
from scipy.special import logit, expit
import matplotlib.pyplot as plt

from arpym.statistics import meancov_sp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_fda-parameters)

n_samples = 5000
mu_z = np.zeros(2)  # expectation
sigma2_z = np.array([[1, 0], [0, 1]])  # covariance

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_fda-implementation-step01): Define features and target variables

# +

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)


def sigf(z1, z2):
    return np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))


z = np.random.multivariate_normal(mu_z, sigma2_z, n_samples)
psi_z = np.c_[z[:, 0], z[:, 0]*z[:, 1]]
x = muf(z[:, 0], z[:, 1]) +\
       sigf(z[:, 0], z[:, 1]) * np.random.randn(n_samples)

x = np.heaviside(x, 1)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_fda-implementation-step02): Fisher discriminant analysis

# +
e_z0, cv_z0 = meancov_sp(psi_z[x==0])
e_z1, cv_z1 = meancov_sp(psi_z[x==1])
p = len(x[x==1])/n_samples
e_cv_zx = (1-p)*cv_z0 + p*cv_z1

gamma = np.linalg.norm(np.linalg.solve(e_cv_zx, e_z1 - e_z0))
beta = (1/gamma)*np.linalg.solve(e_cv_zx, e_z1 - e_z0)
alpha = (1/gamma)*logit(p) - beta@(e_z1 + e_z0)/2

def chi_alphabeta(z):
    return np.heaviside(alpha + z@beta.T, 1)



# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_fda-implementation-step05): Predictions

# +
x_pred = chi_alphabeta(psi_z)

# Probabilities
p_class = expit(alpha + z@beta.T)

# Scores
s_0 = alpha + z[x == 0]@beta.T
s_1 = alpha + z[x == 1]@beta.T

# ROC curve and AUC
fpr_class, tpr_class, _ = roc_curve(x, p_class)
auc_class = auc(fpr_class, tpr_class)

# Error
err = np.mean((x-x_pred)**2)

# Confusion matrix
cm_class = confusion_matrix(x, x_pred)
cm_class = cm_class/np.sum(cm_class, axis=1)
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()

# Parameters
orange = [0.94, 0.35, 0]
green = [0, 0.5, 0]
grtrue = [0, 0.4, 0]
grhatch = [0, 0.1, 0]
grfalse = [0.2, 0.6, 0]
n_classes = 2
plot_colors = "rb"
plot_step = 0.02
psi_zz1, psi_zz2 = np.meshgrid(np.arange(-2, 2, plot_step),
                               np.arange(-2, 2, plot_step))
idxx0 = np.where(np.abs(psi_z[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(psi_z[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)

# ROC curve
ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
ax1.plot(fpr_class, tpr_class, color='b')
ax1.set_xlabel('fpr')
ax1.set_ylabel('tpr')
ax1.text(0.6, 0., 'AUC = %.2f' % auc_class)
plt.text(0.6, 0.2, 'Error = %.2f' % err)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.axis('square')
ax1.set_title('ROC curve', fontweight='bold')

# 3D plot
ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2, projection='3d')
x_plot = chi_alphabeta(np.c_[psi_zz1.ravel(), psi_zz2.ravel()])
x_plot = x_plot.reshape(psi_zz1.shape)
ax2.plot_surface(psi_zz1, psi_zz2, x_plot,
                 cmap=plt.cm.RdYlBu, alpha=0.7)

# Scatter plot
for i, color in zip(range(n_classes), plot_colors):
    idx = np.intersect1d(np.where(x == i), idxx)
    ax2.scatter3D(psi_z[idx, 0], psi_z[idx, 1], x[idx], c=color,
                  label=['0', '1'][i],
                  cmap=plt.cm.RdYlBu, s=10, alpha=1)
ax2.view_init(30, -90)
ax2.set_xlabel('$\psi_1(Z)$')
ax2.set_ylabel('$\psi_2(Z)$')
ax2.set_zlabel('$X$')
ax2.set_title('Surface fitted with FDA classifier', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
add_logo(fig, axis=ax2, location=2, size_frac_x=1/8)
plt.tight_layout()

# Regions plot
ax3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)
xx_pred = chi_alphabeta(np.c_[psi_zz1.ravel(), psi_zz2.ravel()])
# Put the result into a color plot
xx_pred = xx_pred.reshape(psi_zz1.shape)
ax3.contourf(psi_zz1, psi_zz2, xx_pred, cmap=plt.cm.RdYlBu, alpha=0.5)
# Scatter plot
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(x == i)
    ax3.scatter(psi_z[idx, 0], psi_z[idx, 1], c=color,
                label=['0', '1'][i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15, alpha=0.7)
ax3.set_xlabel('$\psi_1(Z)$')
ax3.set_ylabel('$\psi_2(Z)$')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
ax3.set_title('Decision regions', fontweight='bold')

# Scores
ax4 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=1)
ax4.hist(s_0, 80, density=True, alpha=0.7, color='r')
ax4.hist(s_1, 80, density=True, alpha=0.7, color='b')
yymax = ax4.get_ylim()[1]
ax4.plot([0, 0], [0, yymax], 'k--')
ax4.legend(['S | 0', 'S | 1'])
ax4.set_title('Scores distribution', fontweight='bold')

# Rates
ax5 = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=1)
ax5.fill([0.15, 0.15, 0.35, 0.35],
         [0, cm_class[0, 1], cm_class[0, 1], 0], facecolor=grfalse,
         edgecolor=grhatch, hatch='//', alpha=0.7)
ax5.fill([0.15, 0.15, 0.35, 0.35],
         [cm_class[0, 1], 1, 1, cm_class[0, 1]],
         facecolor=grfalse, alpha=0.7)
ax5.fill([0.45, 0.45, 0.65, 0.65],
         [0, cm_class[1, 1], cm_class[1, 1], 0], facecolor=grtrue,
         alpha=0.7)
ax5.fill([0.45, 0.45, 0.65, 0.65],
         [cm_class[1, 1], 1, 1, cm_class[1, 1]], facecolor=grtrue,
         edgecolor=grhatch, hatch='\\\\', alpha=0.7)
ax5.set_ylim([0, 1])
ax5.legend(['fpr', 'tnr', 'tpr', 'fnr'], bbox_to_anchor=(0.001, -0.07, 1., .1),
           facecolor='white',
           loc=1, ncol=5, mode="expand")
ax5.set_title('Confusion matrix', fontweight='bold')
ax5.set_xticks([])
ax5.grid(False)

plt.tight_layout()

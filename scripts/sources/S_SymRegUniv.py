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

# # S_SymRegUniv [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_SymRegUniv&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sym-reg).

# ## Prepare the environment

# +
import os.path as path, sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from pcacov import pcacov
from RawMigrationDb2AggrRiskDrivers import RawMigrationDb2AggrRiskDrivers

# input parameters
n_ = 1  # dimension of target variable X
k_ = 1  # dimension of factor Z
mu_XZ = array([2, 1])  # joint expectation of target X and factor Z
sigma2_XZ = array([[3, 2.1], [2.1, 2]])  # joint covariance of target X and factor Z
# -

# ## Compute the linear regression loadings

beta_Reg_XZ = sigma2_XZ[0, 1] / sigma2_XZ[1, 1]
beta_Reg_ZX = sigma2_XZ[1, 0] / sigma2_XZ[0, 0]

# ## Compute the symmetric regression loadings

# +
e, _ = pcacov(sigma2_XZ)

beta_Sym_XZ = -e[1, 1] / e[0, 1]
beta_Sym_ZX = 1 / (-e[1, 1] / e[0, 1])
# -

# ## Compute the parameters of the symmetric regression recovered target

alpha_Sym_XZ = mu_XZ[0] - beta_Sym_XZ * mu_XZ[1]
mu_X_tilde_Sym = alpha_Sym_XZ + beta_Sym_XZ * mu_XZ[1]
sigma2_X_tilde_Sym = beta_Sym_XZ ** 2 * sigma2_XZ[1, 1]

# ## Compute the parameters of the symmetric regression recovered factor

alpha_Sym_ZX = mu_XZ[1] - beta_Sym_ZX * mu_XZ[0]
mu_Z_tilde_Sym = alpha_Sym_ZX + beta_Sym_ZX * mu_XZ[0]
sigma2_Z_tilde_Sym = beta_Sym_ZX ** 2 * sigma2_XZ[0, 0]

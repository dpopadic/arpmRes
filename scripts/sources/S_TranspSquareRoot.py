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

# # S_TranspSquareRoot [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_TranspSquareRoot&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-exer-cpca-copy-4).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from TransposeSquareRoot import TransposeSquareRoot

# symmetric and positive definite matrix
sigma2 = array([[0.25, 0.30, 0.25], [0.30, 1, 0], [0.25, 0, 6.25]])
# -

# ## Compute the transpose-square-root matrix s of sigma2 by:

# +
# i) Riccati root
s_riccati = TransposeSquareRoot(sigma2, 'Riccati')

# ii) Conditional Principal Component Analysis (CPCA)
# full rank linear constraints matrix
d = array([[1, 0, 1], [0, 1, 0]])
s_cpca = TransposeSquareRoot(sigma2, 'CPCA', d)

# iii) Principal Component Analysis (PCA)
s_pca = TransposeSquareRoot(sigma2, 'PCA')

# iv) Cholesky decomposition using LDL decomposition
s_chol = TransposeSquareRoot(sigma2, 'Chol')

# v) Gram-Schmidt process
s_gs = TransposeSquareRoot(sigma2, 'Gram-Schmidt')
# -

# ## For each method check that sigma2 = s@s.T holds true

check_cpca = s_cpca@s_cpca.T
check_pca = s_pca@s_pca.T
check_riccati = s_riccati@s_riccati.T
check_chol = s_chol@s_chol.T
check_gs = s_gs@s_gs.T

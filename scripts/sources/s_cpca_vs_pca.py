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

# # s_cpca_vs_pca [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cpca_vs_pca&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-exer-cpca).

import numpy as np
from arpym.tools import cpca_cov, pca_cov

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-parameters)

# +
# symmetric and positive definite covariance matrix
sigma2 = np.array([[0.25, 0.30, 0.25], [0.30, 1, 0], [0.25, 0, 6.25]])

# full rank linear constraints matrix
d = np.array([[1, 0, 1], [0, 1, 0]])

k_, n_ = np.shape(d)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-implementation-step01): Compute the conditional principal variances/directions of sigma2

e_d, lambda2_d = cpca_cov(sigma2, d)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-implementation-step02): Compute the product e_d'*sigma2*e_d and check that it coincides with the diagonal matrix Diag(lambda2_d)

err_cpca_diag = np.linalg.norm(e_d.T@sigma2@e_d - np.diag(lambda2_d))/np.linalg.norm(np.diag(lambda2_d))

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-implementation-step03): Compute the products e_d'*e_d and e_d*e_d' and verify that the conditional principal directions are not orthogonal

err_cpca_orth1 = np.linalg.norm(e_d.T@e_d-np.eye(n_))/np.linalg.norm(np.eye(n_))
err_cpca_orth2 = np.linalg.norm(e_d@e_d.T-np.eye(n_))/np.linalg.norm(np.eye(n_))

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-implementation-step04): Compute the principal variances/directions of sigma2

e, lambda2 = pca_cov(sigma2)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-implementation-step05): Compute the product e'*sigma2*e and check that it coincides with the diagonal matrix Diag(lambda2)

err_pca_diag = np.linalg.norm(e.T@sigma2@e - np.diag(lambda2))/np.linalg.norm(np.diag(lambda2))

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_cpca_vs_pca-implementation-step06): Compute the products e'*e and e*e' and verify that the principal directions are orthogonal

err_pca_orth1 = np.linalg.norm(e.T@e-np.eye(n_))/np.linalg.norm(np.eye(n_))
err_pca_orth2 = np.linalg.norm(e@e.T-np.eye(n_))/np.linalg.norm(np.eye(n_))

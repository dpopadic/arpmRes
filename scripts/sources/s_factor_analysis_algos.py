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

# # s_factor_analysis_algos [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_factor_analysis_algos&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-factoranalysis-algos).

# +
import numpy as np

from arpym.estimation import factor_analysis_mlf, factor_analysis_paf
from arpym.views import rel_entropy_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_algos-parameters)

# positive definite matrix
sigma2 = np.array([[1.20, 0.46, 0.77],
                   [0.46, 2.31, 0.08],
                   [0.77, 0.08, 0.98]])
k_ = 1  # number of columns

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_algos-implementation-step01): Compute PAF decomposition of sigma2

beta_paf, delta2_paf = factor_analysis_paf(sigma2, k_=k_)
sigma2_paf = beta_paf @ beta_paf.T + np.diagflat(delta2_paf)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_algos-implementation-step02): Compute MLF decomposition of sigma2

beta_mlf, delta2_mlf = factor_analysis_mlf(sigma2,
                                           k_=k_, b=beta_paf, v=delta2_paf)
sigma2_mlf = beta_mlf @ beta_mlf.T + np.diagflat(delta2_mlf)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_algos-implementation-step03): Compute Frobenius and relative entropy error

err_paf_frobenius = np.linalg.norm(sigma2-sigma2_paf, ord='fro')
err_mlf_frobenius = np.linalg.norm(sigma2-sigma2_mlf, ord='fro')
mean = np.array(np.zeros(sigma2.shape[0]))
err_paf_rel_entr = rel_entropy_normal(mean, sigma2_paf, mean, sigma2)
err_mlf_rel_entr = rel_entropy_normal(mean, sigma2_mlf, mean, sigma2)

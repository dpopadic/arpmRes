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

# # s_cross_section_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cross_section_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmcross-cor).

import numpy as np
import pandas as pd

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-parameters)

beta = np.array([1., 1.]).reshape(-1, 1)  # loadings
k_ = beta.shape[1]  # factor dimension
e = np.array([.5])  # exposure to the loadings

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step01): Load data

# +
path = '../../../databases/temporary-databases'
df = pd.read_csv(path + '/db_pricing_zcb.csv', header=0)

# number of instruments
n_ = len(np.array(df['v_zcb_tnow'].dropna(axis=0, how='all')))
# expectation of target variable
mu_x = np.array(df['mu_pl'].dropna(axis=0, how='all'))
# covariance of target variable
sigma2_x = np.array(df['sig2_pl'].dropna(axis=0, how='all')).reshape((n_, n_))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step02): Compute construction matrix, projector matrix and shift term

sigma2 = np.array([[sigma2_x[0, 0], 0], [0, sigma2_x[1, 1]]])  # scale matrix
beta_ = np.linalg.solve(sigma2, beta)
gamma = np.linalg.solve(beta.T @ beta_, beta_.T)
beta_betainv = beta @ gamma
alpha = mu_x - beta_betainv @ mu_x

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step03): Compute distribution of prediction

m_xbar_cs = mu_x
sigma2_xbar_cs = beta_betainv @ sigma2_x @ beta_betainv.T

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step04): Compute r-squared

r2_sigma2 = np.trace(np.linalg.solve(sigma2, beta_betainv @ sigma2_x)) / \
     np.trace(np.linalg.solve(sigma2, sigma2_x))

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step05): Compute joint distribution of residuals and factor

a = np.concatenate((-alpha, np.zeros(k_)))
b = np.concatenate((np.eye(n_) - beta_betainv, gamma))
m_uz = a + b @ mu_x
sigma2_uz = b @ sigma2_x @ b.T

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step06): Compute optimal parameters

beta_ = np.linalg.solve(sigma2_x, beta)
betainv_mv = np.linalg.solve(beta.T @ beta_, beta_.T)
beta_betainv_mv = beta @ betainv_mv
alpha_mv = mu_x - beta_betainv_mv @ mu_x

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step07): Compute the intuitive r-squared

r2_sigma2_mv = np.trace(beta_betainv_mv) / n_

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step08): Compute the regression loadings

beta_reg = np.linalg.solve(gamma @ sigma2_x @ gamma.T, gamma @ sigma2_x).T

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step09): Compute the regression loadings of the optimal construction

beta_reg_mv = np.linalg.solve(betainv_mv @ sigma2_x @ betainv_mv.T,
                               betainv_mv @ sigma2_x).T

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step10): Compute the joint distribution based on optimal construction

a = np.concatenate((-alpha_mv, np.zeros(k_)))
b = np.concatenate((np.eye(n_) - beta_betainv_mv, betainv_mv))
m_uz_mv = a + b @ mu_x
sigma_2_uz_mv = b @ sigma2_x @ b.T

# ## [Step 11](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step11): Compute the minimum variance combination

h_mv = e @ betainv_mv

# ## [Step 12](https://www.arpm.co/lab/redirect.php?permalink=s_cross_section_lfm-implementation-step12): Save database

output = {'k_': pd.Series(k_),
          'beta': pd.Series(beta.reshape(-1, )),
          'alpha': pd.Series(alpha)
          }
df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_cross_section.csv')



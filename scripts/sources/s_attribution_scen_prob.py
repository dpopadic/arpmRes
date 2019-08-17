#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_attribution_scen_prob [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_attribution_scen_prob&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBTopDownExpSP).

# +
import numpy as np
import pandas as pd

from arpym.statistics import meancov_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_scen_prob-parameters)

k_ = 2  # number of risk factors
rh_z = np.array([[-0.01057143, -0.0041252, -0.01986819],
               [-0.02405714, -0.00980853, 0.01450357],
               [0.00657143, -0.00406089, 0.01188747],
               [0.01925714, 0.02680999, 0.00541017]])   # scenario realizations
p = np.array([0.3, 0.1, 0.4, 0.2])  # probabilities
j_ = p.shape[0]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_scen_prob-implementation-step01): Top-down exposures

m_rh_z, s_rh_z = meancov_sp(rh_z, p)  # scenario-probability mean and covariance
# top-down exposures
beta = s_rh_z[0, 1:]@(np.linalg.solve(s_rh_z[1:, 1:], np.eye(k_)))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_scen_prob-implementation-step02): Shift term

alpha = m_rh_z[0] - beta@m_rh_z[1:]

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_scen_prob-implementation-step03): Scenarios for the residuals

u = rh_z[:, 0] - alpha - beta@rh_z[:, 1:].T

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_scen_prob-implementation-step04): Joint scenarios for U,Z

uz = np.r_['-1', u[np.newaxis, ...].T, rh_z[:, 1:3]]

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_scen_prob-implementation-step05): Save the data

# +
output = {'k_': pd.Series(k_),
          'j_': pd.Series(j_),
          'p': pd.Series(p),
          'beta': pd.Series(beta),
          'alpha': pd.Series(alpha),
          'rh_z': pd.Series(rh_z.reshape((j_*(k_+1),))),
          'uz': pd.Series(uz.reshape((j_*(k_+1),)))}

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_attribution_scen_prob.csv',
          index=None)

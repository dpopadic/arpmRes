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

# # s_risk_attribution_scen_prob [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_risk_attribution_scen_prob&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerciseScenProbRiskAttr).

# +
import numpy as np
import pandas as pd
from scipy.stats import norm

from arpym.statistics import meancov_sp

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-parameters)

c = 0.99  # quantile and cVaR confidence level
zeta = 2  # Esscher parameter
theta = -0.1  # parameter for Wang expectation
alpha_ph = 0.5  # parameter for proportional hazards expectation

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step01): Load data

path = '../../../databases/temporary-databases/'
db = pd.read_csv(path + 'db_attribution_scen_prob.csv')
k_ = pd.read_csv('../../../databases/temporary-databases/db_attribution_scen_prob.csv',
                 usecols=['k_'], nrows=1).values[0, 0].astype(int)
j_ = pd.read_csv('../../../databases/temporary-databases/db_attribution_scen_prob.csv',
                 usecols=['j_'], nrows=1).values[0, 0].astype(int)
p = np.array(db['p'].iloc[:j_]).reshape(-1)  # probabilities
alpha = np.array(db['alpha'].iloc[0])  # shift term
beta = np.array(db['beta'].iloc[:j_ - 2]).reshape(-1, 1)  # top-down exposures
# scenario realizations of ex-ante performance and factors
rh_z = np.array(db['rh_z'].iloc[:j_ * (k_ + 1)]). \
    reshape((j_, k_ + 1))
# scenario realizations of residual and factors
uz = np.array(db['uz'].iloc[:j_ * (k_ + 1)]). \
    reshape((j_, k_ + 1))
pi_b_resc = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                        usecols=['pi_b_resc']).iloc[:j_].values.reshape(j_, )

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step02): Compute the scenarios for the risk factor Z_0 and update the exposures

# +
beta_0 = 1  # exposure to the residual
beta = np.append(beta_0, beta)  # updated exposures

u = uz[:, 0]  # scenarios of the residual
z0 = (alpha + u)  # scenarios for the risk factor Z_0
z = np.r_['-1', z0.reshape(j_, 1), uz[:, 1:]]  # update risk factors
_, cov_z = meancov_sp(z, p)  # covariance of the factors

r_h = rh_z[:, 0]  # ex-ante performance scenarios
_, var_r_h = meancov_sp(r_h, p)  # variance of the ex-ante performance
sd_r_h = np.sqrt(var_r_h)  # standard deviations of the ex-ante performance
satis_r_h = -sd_r_h  # total satisfaction st.dev.
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step03): Computation of the first-in/isolated proportional attribution

satis_bkzk = -np.abs(beta) * np.sqrt(np.diag(cov_z)).T
gamma_isol = satis_r_h / np.sum(satis_bkzk)  # normalization constant
satis_k_isol = gamma_isol * satis_bkzk  # "first in" proportional contributions

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step04): Computation of the last-in proportional attribution

satis_rhminusbkzk = -np.sqrt(satis_r_h ** 2 + (beta * beta) *
                             np.diag(cov_z).T - 2 * beta * (beta @ cov_z))
satis_diff = satis_r_h - satis_rhminusbkzk  # yet to be rescaled
gamma_last = satis_r_h / np.sum(satis_diff)  # normalization constant
satis_k_last = gamma_last * satis_diff  # "last in" prop. contributions

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step05): Computation of the sequential risk attribution

# +
index = [1, 2, 0]
beta_reshuf = beta[index]  # reshuffled exposures
cov_z_reshuf = cov_z[:, index][index]  # reshuffled factors covariance

satis_up_to = np.zeros(k_ + 2)
for k in range(1, k_ + 2):
    # sum of satisfaction up to k
    satis_up_to[k] = -np.sqrt(beta_reshuf[:k] @
                              cov_z_reshuf[:k, :k] @
                              beta_reshuf[:k].T)

satis_k_seq = np.diff(satis_up_to)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step06): Shapley risk attribution

# +
satis_k_shapley = np.zeros(k_ + 1)

c_ = 4  # number of coalitions

# k=0; coalitions: {0}, {0,1}, {0,2}, {0,1,2}
satis_0 = np.zeros(c_)
# compute contribute for each coalition
satis_0[0] = (1 / 3) * satis_bkzk[0]
satis_0[1] = (1 / 6) * (-np.sqrt(beta[[1, 0]] @ cov_z[np.ix_([1, 0], [1, 0])] @
                                 beta[[1, 0]].T) - satis_bkzk[1])
satis_0[2] = (1 / 6) * (-np.sqrt(beta[[2, 0]] @ cov_z[np.ix_([2, 0], [2, 0])] @
                                 beta[[2, 0]].T) - satis_bkzk[2])
satis_0[3] = (1 / 3) * (-np.sqrt(beta[[1, 2, 0]] @
                                 cov_z[np.ix_([1, 2, 0], [1, 2, 0])] @
                                 beta[[1, 2, 0]].T) +
                        np.sqrt(beta[[1, 2]] @ cov_z[np.ix_([1, 2], [1, 2])] @
                                beta[[1, 2]].T))

# sum over coalitions
satis_k_shapley[0] = np.sum(satis_0)  # yet to be rescaled

# k=1; coalitions: {1}, {0, 1}, {1, 2}, {0, 1, 2}
satis_1 = np.zeros(c_)
# compute contribute for each coalition
satis_1[0] = (1 / 3) * satis_bkzk[1]
satis_1[1] = (1 / 6) * (-np.sqrt(beta[[0, 1]] @ cov_z[np.ix_([0, 1], [0, 1])] @
                                 beta[[0, 1]].T) - satis_bkzk[0])
satis_1[2] = (1 / 6) * (-np.sqrt(beta[[2, 1]] @ cov_z[np.ix_([2, 1], [2, 1])] @
                                 beta[[2, 1]].T) - satis_bkzk[2])
satis_1[3] = (1 / 3) * (-np.sqrt(beta[[0, 2, 1]] @
                                 cov_z[np.ix_([0, 2, 1], [0, 2, 1])] @
                                 beta[[0, 2, 1]].T) +
                        np.sqrt(beta[[0, 2]] @ cov_z[np.ix_([0, 2], [0, 2])] @
                                beta[[0, 2]].T))

# sum over coalitions
satis_k_shapley[1] = np.sum(satis_1)  # yet to be rescaled

# k=2; coalitions: {2}, {0, 2}, {1, 2}, {0, 1, 2}
satis_2 = np.zeros(c_)
# compute contribute for each coalition
satis_2[0] = (1 / 3) * satis_bkzk[2]
satis_2[1] = (1 / 6) * (-np.sqrt(beta[[0, 2]] @ cov_z[np.ix_([0, 2], [0, 2])] @
                                 beta[[0, 2]].T) - satis_bkzk[0])
satis_2[2] = (1 / 6) * (-np.sqrt(beta[[1, 2]] @ cov_z[np.ix_([1, 2], [1, 2])] @
                                 beta[[1, 2]].T) - satis_bkzk[1])
satis_2[3] = (1 / 3) * (-np.sqrt(beta[[0, 1, 2]] @
                                 cov_z[np.ix_([0, 1, 2], [0, 1, 2])] @
                                 beta[[0, 1, 2]].T) +
                        np.sqrt(beta[[0, 1]] @ cov_z[np.ix_([0, 1], [0, 1])] @
                                beta[[0, 1]].T))

# sum over coalitions
satis_k_shapley[2] = np.sum(satis_2)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step07): Computation of the risk marginal contributions: standard deviations

sd_satis_k_euler = np.zeros(k_ + 1)
for k in range(k_ + 1):
    sd_satis_k_euler[k] = beta[k] * ((cov_z @ beta)[k] / np.sqrt(beta @ cov_z @ beta.T))

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step08): Computation of the risk marginal contributions: variance

v_satis_k_euler = np.zeros(k_ + 1)
for k in range(k_ + 1):
    v_satis_k_euler[k] = beta[k] * (cov_z @ beta)[k]

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step08): Sorting

r_h_sort = np.sort(r_h)
index = np.argsort(r_h)
z_sort = z[index, :]
p_sort = p[index]

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step09): Wang

f_wang = norm.cdf(norm.ppf(np.cumsum(p_sort)) - theta)
w_wang_spectr = np.append(f_wang[0], np.diff(f_wang))
wang_k = np.zeros(k_ + 1)
for k in range(k_ + 1):
    wang_k[k] = beta[k] * (z_sort[:, k] @ w_wang_spectr)  # marg. contributions

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step10): Proportional hazard

f_prop_haz = (np.cumsum(p_sort)) ** alpha_ph  # proportional hazards transform
w_prop_haz_spectr = np.append(f_prop_haz[0], np.diff(f_prop_haz))  # derivative
prop_haz_k = np.zeros(k_ + 1)
for k in range(k_ + 1):
    prop_haz_k[k] = beta[k] * (z_sort[:, k] @ w_prop_haz_spectr)  # marg. contributions

# ## [Step 11](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step11): Expected shortfall

# +
u_sort = np.r_[0, np.cumsum(p_sort)]  # cumulative sum of ordered probs.
j_c = next(i for i, x in enumerate(u_sort) if x <= 1 - c)

es_k = np.zeros(k_ + 1)
for k in range(k_ + 1):
    es_k[k] = beta[k] * np.sum(z_sort[:j_c + 1, k]) / (1 - c)  # marg. contributions
# -

# ## [Step 12](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attribution_scen_prob-implementation-step12): Computation of the risk marginal contributions: Esscher expectation and Bulhmann expectation

# +
esscher_exp_k = np.zeros(k_ + 1)
for k in range(k_ + 1):
    # marginal contributions
    esscher_exp_k[k] = beta[k] * \
                       meancov_sp(np.exp(-zeta * r_h) * z[:, k], p)[0] / \
                       meancov_sp(np.exp(-zeta * r_h), p)[0]

bulhmann_expectation_r_h_k = np.zeros(k_ + 1)
for k in range(k_ + 1):
    bulhmann_expectation_r_h_k[k] = beta[k] * meancov_sp(np.exp(-zeta * pi_b_resc) * z[:, k], p)[0] \
                                    / meancov_sp(np.exp(-zeta * pi_b_resc), p)[0]

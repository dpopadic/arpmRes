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

# # s_evaluation_satis_scenprob [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_satis_scenprob&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBEvalNumericalExample).

# +
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from arpym.portfolio import spectral_index
from arpym.statistics import meancov_sp, quantile_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-parameters)

c = 0.99  # confidence level
lam_1 = 0.5  # parameter for mean-variance and mean-semideviation trade-off
lam_2 = 2  # parameter for certainty-equivalent (exponential function)
alpha = 0.25  # parameter for α-expectile
zeta = 2  # parameter for Esscher expectation
theta = -0.1  # parameter for Wang expectation
alpha_ph = 0.5  # parameter for proportional hazards expectation
r = 0.0001  # target for omega ratio
z = np.array([-0.0041252, -0.00980853,  -0.00406089,  0.02680999])  # risk factor

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step01): Load data from db_aggregation_scenario_numerical

j_ = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                 usecols=['j_'], nrows=1).values[0, 0].astype(int)
n_ = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                 usecols=['n_'], nrows=1).values[0, 0].astype(int)
pi = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['pi']).values.reshape(j_, n_)
p = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['p']).iloc[:j_].values.reshape(j_, )
v_h = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                 usecols=['v_h'], nrows=1).values[0, 0].astype(int)
v_b = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                 usecols=['v_b'], nrows=1).values[0, 0].astype(int)
h = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['h']).iloc[:n_].values.reshape(n_, )
h_b = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['h_b']).iloc[:n_].values.reshape(n_, )
h_tilde = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['h_tilde']).iloc[:n_].values.reshape(n_, )
r_h = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['r_h']).iloc[:j_].values.reshape(j_, )
pi_b_resc = pd.read_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv',
                usecols=['pi_b_resc']).iloc[:j_].values.reshape(j_, )

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step02): Compute the expectation, variance, standard deviation of the ex-ante performance

mu_r_h, s2_r_h = meancov_sp(r_h, p)  # ex-ante performance exp. and var.
std_r_h = np.sqrt(s2_r_h)  # standard deviation
s2_satis = - s2_r_h  # variance index of satisfaction
std_satis = -std_r_h  # standard deviation index of satisfaction

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step03): Compute mean-variance trade-off, expectation and covariance of the instruments P&L's, and then the mean-variance trade-off

mv_r_h = mu_r_h - lam_1 / 2 * s2_r_h  # mean-variance trade-off (definition)
mu_pi, s2_pi = meancov_sp(pi, p)  # instruments P&L's exp. and cov.
mv_r_h_1 = h_tilde@mu_pi - (lam_1 / 2) * h_tilde@s2_pi@h_tilde # mean-variance trade-off (quadratic form)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step04): Compute the certainty-equivalent

# +
# exponential utility function
def exp_utility_f(y, lam):
    return -np.exp(-lam * y)


# inverse exponential utility function
def ce_f(z, lam):
    return -(1 / lam) * np.log(-z)


utility_r_h = exp_utility_f(r_h, lam_2)  # utility
mu_utility = utility_r_h@p  # expected utility computation
cert_eq_r_h = ce_f(mu_utility, lam_2)  # certainty-equivalent]
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step05): Compute the quantile (VaR)

q_r_h = quantile_sp(1-c, r_h, p=p)


# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step06): Compute the Expected shortfall

# +
# compute expected shortfall using spectral_index

# indicator function
def indicator(x):
    return (0 <= x and x <= 1-c)

# spectrum function
def spectr_es(x):
    return (1 / (1 - c)) * indicator(x)

# negative expected shortfall
es2, _ = spectral_index(spectr_es, pi, p, h_tilde)

r_h_sort = np.sort(r_h)
index = np.argsort(r_h)
p_sort = p[index]

u_sort = np.r_[0, np.cumsum(p_sort)]  # cumulative sum of ordered probs.
j_c = next(i for i, x in enumerate(u_sort) if 0 <= x and x <= 1-c)

es = np.sum(r_h_sort[:j_c+1])/(1-c)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step07): Compute the Wang expectation

f_wang = norm.cdf(norm.ppf(np.cumsum(p_sort)) - theta)
w_wang_spectr = np.append(f_wang[0], np.diff(f_wang))
wang_expectation_r_h = r_h_sort@w_wang_spectr

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step08): Compute the proportional hazard expectation

f_prop_haz = (np.cumsum(p_sort)) ** alpha_ph  # proportional hazards transform
w_prop_haz_spectr = np.append(f_prop_haz[0], np.diff(f_prop_haz))  # derivative
# ex-ante performance proportional hazards expectation
prop_haz_expectation_r_h = r_h_sort@w_prop_haz_spectr

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step09): Compute the mean-semideviation trade-off

semiv_r_h = sum(((r_h[r_h <= mu_r_h] - mu_r_h) ** 2)
                * p[r_h <= mu_r_h])  # ex-ante performance semivariance
semid_r_h = (semiv_r_h) ** (0.5)  # ex-ante performance semideviation
# ex-ante performance mean-semideviation trade-off
msemid_r_h = mu_r_h - lam_1 * semid_r_h

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step10): Compute the alpha-expectile

# +
def expectile_f(x, p, alpha):
    return alpha * np.sum(p * np.maximum(r_h - x, 0)) + \
        (1 - alpha) * (np.sum(p * np.minimum(r_h - x, 0)))


# ex-ante performance α-expectile
expectile_r_h = fsolve(expectile_f, -0.01, args=(p, alpha))
# -

# ## [Step 11](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step11): Compute information ratio, Sortino ratio and omega ratio

info_ratio_r_h = mu_r_h / std_r_h  # ex-ante performance information ratio
# ex-ante performance Sortino ratio
sortino_ratio_r_h = (mu_r_h - r) / np.sqrt((np.maximum(r - r_h, 0) ** 2)@p)
# ex-ante performance omega ratio
omega_ratio_r_h = (np.maximum(r_h - r, 0)@p) / (np.maximum(r - r_h, 0)@p)
omega_ratio_1_r_h = (r_h@p - r) / (np.maximum(r - r_h, 0)@p) + 1

# ## [Step 12](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step12): Compute the scenario-probability distribution of factor Z, beta, correlation

mu_z, s2_z = meancov_sp(z, p)  # variance of z
cv_yz = (r_h * z)@p - mu_r_h * mu_z  # covariance of r_h and z
beta_r_h_z = - cv_yz / s2_z  # ex-ante performance opposite of beta
# opposite of correlation between performance and factor
corr_r_h_z = - cv_yz / (np.sqrt(s2_r_h) * np.sqrt(s2_z))

# ## [Step 13](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step13): Compute the Buhlmann expectation and the Esscher expectation

# +
bulhmann_expectation_r_h, _ = meancov_sp(np.exp(-zeta * pi_b_resc) * r_h, p)[0] \
    / meancov_sp(np.exp(-zeta * pi_b_resc), p)

esscher_expectation_r_h, _ = meancov_sp(np.exp(-zeta * r_h) *
                                        r_h, p)[0] \
    / meancov_sp(np.exp(-zeta * r_h), p)
# -

# ## [Step 14](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step14): Save the data

# +
output = {'s2_satis': pd.Series(s2_satis),
          'std_satis': pd.Series(std_satis),
          'wang_expectation_r_h': pd.Series(wang_expectation_r_h),
          'prop_haz_expectation_r_h': pd.Series(prop_haz_expectation_r_h),
          'expectile_r_h': pd.Series(expectile_r_h),
          'bulhmann_expectation_r_h': pd.Series(bulhmann_expectation_r_h),
          'esscher_expectation_r_h': pd.Series(esscher_expectation_r_h)
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_evaluation_scenprob.csv')

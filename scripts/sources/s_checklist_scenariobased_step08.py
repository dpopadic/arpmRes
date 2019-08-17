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

# # s_checklist_scenariobased_step08 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step08&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-8).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import meancov_sp
from arpym.estimation import fit_lfm_lasso
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step08-parameters)

# +
# indicates which projection to continue from
# True: use copula-marginal projections
# False: use historical projections
copula_marginal = True

# parameter for lasso minimization
if copula_marginal:
    lam = 98000
else:
    lam = 15000
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step08-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

# Risk drivers identification
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series.csv',
                                    index_col=0)
x = db_riskdrivers_series.values
riskdriver_names = np.array(db_riskdrivers_series.columns)

db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
d_ = int(db_riskdrivers_tools['d_'][0])
n_stocks = int(db_riskdrivers_tools['n_stocks'][0])
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')

# Pricing
db_holdings = pd.read_csv(path+'db_holdings.csv')

if copula_marginal:
    # Projection
    db_projection_tools = pd.read_csv(path+'db_projection_tools.csv')
    j_ = int(db_projection_tools['j_'][0])
    t_hor = np.datetime64(db_projection_tools['t_hor'][0], 'D')
    m_ = np.busday_count(t_now, t_hor)

    db_projection_riskdrivers = pd.read_csv(path+'db_projection_riskdrivers.csv')
    x_proj = db_projection_riskdrivers.values.reshape(j_, m_+1, d_)

    db_scenprob = pd.read_csv(path+'db_scenario_probs.csv')
    p = db_scenprob['p'].values

    # Aggregation
    db_exante_perf = pd.read_csv(path+'db_exante_perf.csv')
    y_h = db_exante_perf.values.squeeze()

    # Ex-ante evaluation
    db_quantile_and_satis = pd.read_csv(path+'db_quantile_and_satis.csv')
    c_es = db_quantile_and_satis['c_es'][0]
    es_yh = db_quantile_and_satis['es_yh'][0]
    neg_var_yh = db_quantile_and_satis['neg_var_yh'][0]
else:
    # Projection
    db_projection_tools = pd.read_csv(path+'db_projection_bootstrap_tools.csv')
    j_ = int(db_projection_tools['j_'][0])
    t_hor = np.datetime64(db_projection_tools['t_hor'][0], 'D')
    m_ = np.busday_count(t_now, t_hor)

    db_projection_riskdrivers = pd.read_csv(path+'db_projection_bootstrap_riskdrivers.csv')
    x_proj = db_projection_riskdrivers.values.reshape(j_, m_+1, d_)

    db_scenprob = pd.read_csv(path+'db_scenario_probs_bootstrap.csv')
    p = db_scenprob['p'].values

    # Aggregation
    db_exante_perf = pd.read_csv(path+'db_exante_perf_historical.csv')
    y_h = db_exante_perf.values.squeeze()

    # Ex-ante evaluation
    db_quantile_and_satis = pd.read_csv(path+'db_quantile_and_satis_historical.csv')
    c_es = db_quantile_and_satis['c_es'][0]
    es_yh = db_quantile_and_satis['es_yh'][0]
    neg_var_yh = db_quantile_and_satis['neg_var_yh'][0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step08-implementation-step01): Ex-ante attribution: performance

# +
# risk factors: risk driver increments
z = x_proj[:, -1, :]-x[-1, :]
# estimate exposures, intercept and residuals
alpha, beta, _, u = fit_lfm_lasso(y_h, z, p, lam)
u = u.squeeze()
alpha = alpha[0]

# select data for relevant risk factors only
ind_relevant_risk_factors = np.where(beta != 0)[0]
beta = beta[ind_relevant_risk_factors]
z = z[:, ind_relevant_risk_factors]
# number of relevant risk factors
k_ = beta.shape[0]

# joint distribution of residual and risk factors
f_uz = (np.c_[u, z], p)

risk_factors = riskdriver_names[ind_relevant_risk_factors]
print('Number of relevant risk factors: ' + str(k_))

# create output dictionary
output = {'k_': k_,  # number of relevant risk factors
          'alpha': alpha,  # shift term
          'beta': beta,  # exposures
          'f_UZ': f_uz  # joint distribution of residual and risk factors
          }
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step08-implementation-step02): Ex-ante attribution: risk

# +
# map residuals to 0-th factor
z_0 = (alpha + u)
# exposure to the residual
beta_0 = 1

# update exposures
beta_new = np.append(beta_0, beta)
k_new = beta_new.shape[0]
# update risk factors
z_new = np.c_[z_0, z]

# sort the scenarios of the risk factors and probabilities
# according to order induced by ex-ante performance scenarios
sort_yh = np.argsort(y_h, axis=0)
p_sort = p[sort_yh]
z_new_sort = z_new[sort_yh, :]

# marginal contributions to the negative expected shortfall satisfaction measure
# calculate weights
j_c = np.min(np.where(np.cumsum(p_sort) >= 1-c_es))
w = np.zeros((j_))
for j in range(j_c):
    w[j] = 1/(1-c_es)*p_sort[j]
w[j_c] = 1 - np.sum(w)
# calculate contributions
es_contrib = beta_new * (w.T @ z_new_sort)
# print percentage contributions
pc_es_contrib = es_contrib/np.sum(es_yh)
print('Percentage contributions to negative expected shortfall')
print('-'*55)
for k in range(1, k_+1):
    print('{:31}'.format(risk_factors[k-1])+':',
          '{: 7.2%}'.format(pc_es_contrib[k]))
print('{:31}'.format('residual')+':',
      '{: 7.2%}'.format(pc_es_contrib[0]))
print('')

# marginal contributions to the variance satisfaction measure
# find covariance
_, cov_z_new = meancov_sp(z_new, p)
# calculate contributions
var_contrib = -beta_new * (cov_z_new @ beta_new.T)
# print percentage contributions
pc_var_contrib = var_contrib/neg_var_yh
print('Percentage contributions to variance satisfaction measure')
print('-'*57)
for k in range(1, k_+1):
    print('{:31}'.format(risk_factors[k-1])+':',
          '{: 7.2%}'.format(pc_var_contrib[k]))
print('{:31}'.format('residual')+':',
      '{: 7.2%}'.format(pc_var_contrib[0]))

# update output dictionary
output['-ES_k'] = es_contrib
output['-V_k'] = var_contrib
# -

# ## Plots

# +
plt.style.use('arpm')
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)

# expected shortfall
heights = np.flip(np.append(es_yh, np.append(es_contrib[1:], es_contrib[0])))
heights_r = heights*1e-6
lbls = np.flip(np.append('total', np.append(risk_factors, 'residual')))
colors = ['C5'] + ['C0']*k_ + ['C2']
ax1.barh(range(k_new+1), heights_r,
         tick_label=lbls, color=colors)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax1.set_ylabel('Risk driver increments', fontsize=17)
ax1.set_xlabel('-ES (million USD)', fontsize=17)
ax1.set_title('Risk attribution: expected shortfall',
          fontsize=20, fontweight='bold')

# variance
heights = np.flip(np.append(neg_var_yh, np.append(var_contrib[1:], var_contrib[0])))
colors = ['C5'] + ['C0']*k_ + ['C2']
ax2.barh(range(k_new+1), heights, color=colors)
plt.yticks([])
ax2.set_xlabel('-Variance', fontsize=17)
ax2.set_ylabel('')
ax2.set_title('Risk attribution: variance',
          fontsize=20, fontweight='bold')
plt.tight_layout()

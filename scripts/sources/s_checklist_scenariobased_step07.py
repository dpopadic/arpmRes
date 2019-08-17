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

# # s_checklist_scenariobased_step07 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step07&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-7).

# +
import numpy as np
import pandas as pd

from arpym.portfolio import spectral_index
from arpym.statistics import meancov_sp, quantile_sp

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-parameters)

# +
# indicates which projection to continue from
# True: use copula-marginal projections
# False: use historical projections
copula_marginal = True

lam = 3e-7  # parameter of exponential utility function
c_quantile = 0.95  # confidence level for the quantile satisfaction measure
c_es = 0.95  # confidence level for the negative expected shortfall
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

if copula_marginal:
    # Projection
    db_projection_tools = pd.read_csv(path + 'db_projection_tools.csv')
    j_ = int(db_projection_tools['j_'][0])

    db_scenprob = pd.read_csv(path + 'db_scenario_probs.csv')
    p = db_scenprob['p'].values

    # Pricing
    db_pricing = pd.read_csv(path + 'db_pricing.csv')
    pi_tnow_thor = db_pricing.values

    # Aggregation
    db_exante_perf = pd.read_csv(path + 'db_exante_perf.csv')
    y_h = db_exante_perf.values.squeeze()

else:
    # Projection
    db_projection_tools = pd.read_csv(path + 'db_projection_bootstrap_tools.csv')
    j_ = int(db_projection_tools['j_'][0])

    db_scenprob = pd.read_csv(path + 'db_scenario_probs_bootstrap.csv')
    p = db_scenprob['p'].values

    # Pricing
    db_pricing = pd.read_csv(path + 'db_pricing_historical.csv')
    pi_tnow_thor = db_pricing.values

    # Aggregation
    db_exante_perf = pd.read_csv(path + 'db_exante_perf_historical.csv')
    y_h = db_exante_perf.values.squeeze()

db_holdings = pd.read_csv(path + 'db_holdings.csv')
h = np.squeeze(db_holdings.values)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-implementation-step01): Calculate certainty equivalent satisfaction measure

# +
# expected utility
expected_utility = p @ (-np.exp(-lam * y_h))  # expected utility computation

# certainty equivalent satisfaction measure
cert_eq_yh = -(1 / lam) * np.log(-expected_utility)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-implementation-step02): Quantile satisfaction measure

# quantile
q_yh = quantile_sp(1 - c_quantile, y_h, p, method='kernel_smoothing')


# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-implementation-step03): Expected shortfall satisfaction measure

# +
# indicator function
def indicator(x):
    return (0 <= x and x <= 1 - c_es)


# spectrum function
def spectr_es(x):
    return (1 / (1 - c_es)) * indicator(x)


# negative expected shortfall
es_yh, _ = spectral_index(spectr_es, pi_tnow_thor,
                          p, h)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-implementation-step04): Expectation and variance satisfaction measures

# expectation satisfaction measure
mean_yh, var_yh = meancov_sp(y_h, p)
# opposite of variance is satisfaction measure
neg_var_yh = -var_yh

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step07-implementation-step05): Save database

# +
out = pd.DataFrame({'cert_eq_yh': pd.Series(cert_eq_yh),
                    'q_yh': pd.Series(q_yh),
                    'es_yh': pd.Series(es_yh),
                    'mean_yh': pd.Series(mean_yh),
                    'neg_var_yh': pd.Series(neg_var_yh),
                    'c_es': pd.Series(c_es),
                    'c_quantile': pd.Series(c_quantile)})
if copula_marginal:
    out.to_csv(path + 'db_quantile_and_satis.csv',
               index=False)
else:
    out.to_csv(path + 'db_quantile_and_satis_historical.csv',
               index=False)

del out

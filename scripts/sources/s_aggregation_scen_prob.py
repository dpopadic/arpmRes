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

# # s_aggregation_scen_prob [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_aggregation_scen_prob&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EXAggreRetScenBApproach).

import pandas as pd
import numpy as np

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_scen_prob-parameters)

pi = np.array([[-0.43, -0.19, 0.25, 0.31],
               [0.15, -1.63, -0.05, 0.91]]).T  # joint scenarios
p = np.array([0.3, 0.1, 0.4, 0.2])  # probabilities
h = np.array([2000000, 800000])
v_h = 70*10**6
h_b = np.array([1000000, 1200000])
v_b = 73*10**6

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_scen_prob-implementation-step01): Compute the scenario-probability distribution of the excess return by multiplication

h_tilde_exces = h / v_h - h_b / v_b
excess_r_h_tilde = h_tilde_exces@pi.T

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_scen_prob-implementation-step02): Compute the scenario-probability distribution of the return by multiplication

h_tilde = h / v_h
r_h_tilde = h_tilde@pi.T

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_scen_prob-implementation-step03): Compute the scenario-probability distribution of the excess return by aggregating, rescaling and subtracting scenarios

# +
# aggregation
pi_h = h@pi.T
pi_b = h_b@pi.T

# rescaling
r_h = pi_h / v_h
r_b = pi_b / v_b

# subtraction
excess_r_2nd = r_h - r_b
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_scen_prob-implementation-step04): Save the data

# +
output = {
          'j_': pd.Series(p.shape[0]),
          'n_': pd.Series(pi.shape[1]),
          'excess_r': pd.Series(excess_r_h_tilde),
          'pi_resc': pd.Series(r_h),
          'pi': pd.Series(pi.reshape(-1)),
          'p': pd.Series(p),
          'pi_b_resc': pd.Series(r_b),
          'pi_aggr': pd.Series(pi_h),
          'pi_b_aggr': pd.Series(pi_b),
          'r_h': pd.Series(r_h_tilde),
          'h': pd.Series(h),
          'h_b': pd.Series(h_b),
          'v_h': pd.Series(v_h),
          'v_b': pd.Series(v_b),
          'h_tilde': pd.Series(h_tilde),
          'h_tilde_exces': pd.Series(h_tilde_exces),
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_aggregation_scenario_numerical.csv')

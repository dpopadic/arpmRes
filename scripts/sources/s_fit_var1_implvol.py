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

# # s_fit_var1_implvol [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fit_var1_implvol&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol).

import numpy as np
import pandas as pd
from arpym.estimation import fit_var1

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol-parameters)

tau_select = ['0.164383562', '0.334246575', '0.498630137', '1.0', '2.0']
t_start = '2009-11-02'  # starting date
t_end = '2012-08-31'  # ending date

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol-implementation-step00): Import data

# +
path = '../../../databases/temporary-databases/'

db_riskdrivers = pd.read_csv(path+'db_calloption_rd.csv',
                             index_col=0, parse_dates=True)
db_riskdrivers = db_riskdrivers.loc[t_start:t_end]
dates = pd.to_datetime(np.array(db_riskdrivers.index))[1:]
tau_implvol = np.array([col[col.find(' tau=')+5:]
                        for col in db_riskdrivers.columns])
ind_select = np.in1d(tau_implvol, tau_select)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol-implementation-step01): Compute risk drivers

x = np.log(db_riskdrivers.iloc[:, ind_select].values)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol-implementation-step02): Perform VAR1 fit

b_hat, mu_epsi_hat, _ = fit_var1(x)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol-implementation-step03): Extract invariants realizations

epsi_var1 = x[1:, :] - x[:-1, :]@b_hat.T

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_fit_var1_implvol-implementation-step04): Save database

# +
out = pd.DataFrame({d: epsi_var1[:, d1] for d1, d in enumerate(
        db_riskdrivers.columns.values[ind_select])}, index=dates)

out.index.name = 'dates'
out.to_csv('../../../databases/temporary-databases/db_calloption_epsi_var1.csv',
           columns=db_riskdrivers.columns.values[ind_select])
del out

out = pd.DataFrame({'x_tnow': pd.Series(x[-1,:])})
for i in range(b_hat.shape[0]):
    out = out.join(pd.DataFrame({'b_hat'+str(i): pd.Series(b_hat[:, i])}))
out.to_csv(
          '../../../databases/temporary-databases/db_calloption_var1.csv',
          index=None)
del out

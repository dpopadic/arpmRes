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

# # s_fit_garch_stocks [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fit_garch_stocks&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_fit_garch_stocks).

# +
import numpy as np
import pandas as pd

from arpym.estimation import conditional_fp, exp_decay_fp, fit_garch_fp
from arpym.statistics import meancov_sp, scoring, smoothing
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_fit_garch_stocks-parameters)

tau_hl_garch = 3*252  # half life for GARCH fit
tau_hl_pri = 3*252  # half life for VIX comp. ret. time conditioning
tau_hl_smooth = 4*21  # half life for VIX comp. ret. smoothing
tau_hl_score = 5*21  # half life for VIX comp. ret. scoring
alpha_leeway = 1/4  # probability included in the range centered in z_vix_star

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fit_garch_stocks-implementation-step00): Load data

# +
path_glob = '../../../databases/global-databases/'

# Stocks
db_stocks_sp = pd.read_csv(path_glob +
                           'equities/db_stocks_SP500/db_stocks_sp.csv',
                           header=1, index_col=0, parse_dates=True)
stocks_names = db_stocks_sp.columns.tolist()


# VIX (used for time-state conditioning)
vix_path = path_glob + 'derivatives/db_vix/data.csv'
db_vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                     index_col=0, parse_dates=True)

# intersect dates
dates_rd = pd.DatetimeIndex.intersection(db_stocks_sp.index, db_vix.index)

# update databases
db_stocks_sp = db_stocks_sp.loc[dates_rd, :]
db_vix = db_vix.loc[dates_rd, :]

dates = dates_rd[1:]
t_ = len(dates)

# values
v = db_stocks_sp.values
vix = db_vix.values[:, 0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fit_garch_stocks-implementation-step01): Risk drivers identification

x = np.log(v)  # log-values
d_ = x.shape[1]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fit_garch_stocks-implementation-step02): Quest for invariance

# +
i_ = d_
epsi = np.zeros((t_, i_))
p_garch = exp_decay_fp(t_, tau_hl_garch)

for i in range(i_):
    print('Fitting ' + str(i+1) + '-th GARCH; ' +
          str(int((i+1)/i_*100)) + '% done.')
    _, _, epsi[:, i] = fit_garch_fp(np.diff(x[:, i], axis=0), p_garch)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fit_garch_stocks-implementation-step03): Historical estimation

# +
# time and state conditioning on smoothed and scored VIX returns

# state indicator: VIX compounded return realizations
c_vix = np.diff(np.log(vix))
# smoothing
z_vix = smoothing(c_vix, tau_hl_smooth)
# scoring
z_vix = scoring(z_vix, tau_hl_score)
# target value
z_vix_star = z_vix[-1]
# flexible probabilities
p_pri = exp_decay_fp(len(dates), tau_hl_pri)
p = conditional_fp(z_vix, z_vix_star, alpha_leeway, p_pri)

mu_hat, sig2_hat = meancov_sp(epsi, p)
# -

# ## Save database

# +
out = pd.DataFrame({stocks_names[i]: epsi[:, i]
                    for i in range(i_)}, index=dates)
out = out[list(stocks_names[:i_])]
out.index.name = 'dates'
out.to_csv('../../../databases/temporary-databases/db_fit_garch_stocks_epsi.csv')

out = pd.DataFrame({'mu_hat': pd.Series(mu_hat.reshape(-1)),
                    'sig2_hat': pd.Series(sig2_hat.reshape(-1))})
out.to_csv(
          '../../../databases/temporary-databases/db_fit_garch_stocks_locdisp.csv',
          index=None)

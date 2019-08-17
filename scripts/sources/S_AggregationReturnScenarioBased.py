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

# # S_AggregationReturnScenarioBased [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_AggregationReturnScenarioBased&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBAggrHistoricalExample).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, ones, round, r_

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import struct_to_dict
# -

# ## Load the temporary database generated in script S_PricingScenarioBased, which contains the joint scenario-probability distribution of the instruments' ex-ante P&L's

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_PricingScenarioBased'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_PricingScenarioBased'), squeeze_me=True)

# ## Compute the holdings corresponding to the following portfolio, fully invested in the n_=15 instruments
# ## $50k are invested in the stocks, such that the stocks are equally weighted
# ## The holdings in each bond are h=5000 (i.e. each bond has $5000 notional)
# ## The call options (same expiry, increasing strikes) have respective
# ## holdings equal to [1 -2, 1] (long butterfly strategy)

# +
Stocks = struct_to_dict(db['Stocks'], as_namedtuple=False)
Bonds = struct_to_dict(db['Bonds'], as_namedtuple=False)
Options = struct_to_dict(db['Options'], as_namedtuple=False)
v_tnow = db['v_tnow'].reshape(-1,1)
Pi = db['Pi']
t_ = db['t_']

v0_stocks = 50000
w_stocks = ones((Stocks['n_'], 1)) / Stocks['n_']
Stocks['h'] = round((w_stocks * v0_stocks) / Stocks['v_tnow'].reshape(-1,1))
Bonds['h'] = 5000 * ones((Bonds['n_'], 1))
Options['h'] = array([[1], [-2], [1]])

h = r_[Stocks['h'], Bonds['h'], Options['h']]  # ## holdings
cash = 0
# -

# ### Compute the value of the portfolio and the standardized holdings

vh_tnow = h.T@v_tnow + cash  # portfolio's value
htilde = h / vh_tnow  # standardized holdings

# ### Compute the scenarios of the ex-ante performance (return) distribution

# +
Y_htilde = htilde.T@Pi  # ## ex-ante performance (portfolio return)

sdb = {k:v for k,v in db.items() if not str(k).startswith('__')}
sdb.update({  'Stocks': Stocks,
              'Bonds': Bonds,
              'Options': Options,
              'h': h,
              'cash': cash,
              'vh_tnow': vh_tnow,
              'htilde': htilde,
              'Y_htilde': Y_htilde
                })
savemat(os.path.join(TEMPORARY_DB, 'db_AggregationScenarioBased.mat'), sdb)

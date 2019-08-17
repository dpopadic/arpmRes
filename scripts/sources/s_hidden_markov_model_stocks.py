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

# # s_hidden_markov_model_stocks [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_hidden_markov_model_stocks&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFigHMM).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_hidden_markov_model_stocks-parameters)

name = 'CSCO'  # name of company to consider
t_first = '2007-09-10'  # starting date
t_last = '2012-10-19'  # ending date

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_hidden_markov_model_stocks-implementation-step00): Load data

path = '../../../databases/global-databases/equities/db_stocks_SP500/'
df_stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)
df_stocks = df_stocks.set_index(pd.to_datetime(df_stocks.index))

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_hidden_markov_model_stocks-implementation-step01): Compute the compounded returns

v = df_stocks[name].loc[(df_stocks.index >= t_first) &
                        (df_stocks.index <= t_last)]
dx = np.diff(np.log(v))
dx[np.isnan(dx)] = 0

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_hidden_markov_model_stocks-implementation-step02): Fit the hidden Markov model and get the transaction matrix

# +
hmm = GaussianHMM(n_components=2, means_prior=np.zeros((1, 1)),
                  means_weight=1e10).fit(dx.reshape(-1, 1))

# rearrange the volatility from small to large
sigma2 = hmm.covars_.flatten()
idx = np.argsort(sigma2)
sigma2 = sigma2[idx]
p = hmm.transmat_[np.ix_(idx, idx)]  # transaction matrix
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_hidden_markov_model_stocks-implementation-step03): Compute the hidden status

z_ = hmm.predict(dx.reshape(-1, 1))
z = z_.copy()
z[z_ == 0] = idx[0]
z[z_ == 1] = idx[1]

# ## Plots

# +
plt.style.use('arpm')

panic = dx.copy()
calm = dx.copy()
panic[z == 0] = np.nan
calm[z == 1] = np.nan


fig = plt.figure()
plt.plot(v.index[1:], calm, '.', color=[0.4, 0.4, 0.4])
plt.plot(v.index[1:], panic, '^', color=[1.0, 0.5, 0.0])
plt.legend(['state=calm', 'state=panic'])
plt.ylabel('compound returns')
add_logo(fig)

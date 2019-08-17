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

# # s_fit_discrete_markov_chain [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fit_discrete_markov_chain&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fit-discrete-markov-chain).

# ## Prepare the environment

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as tstu

from arpym.estimation import fit_ratings_markov_chain
from arpym.views import min_rel_entropy_sp
from arpym.statistics import project_trans_matrix
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-parameters)

# +
r = 3 # initial rating
tau_hl = 5  # half-life parameter for credit fit (years)
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step00): Upload data

# +
path = '../../../databases/temporary-databases/'
db_credit = pd.read_csv(path+'db_credit_rd.csv',
                        index_col=0, parse_dates=True)
filt=['(' not in col for col in db_credit.columns]
ratings = [i for indx,i in enumerate(db_credit.columns) if filt[indx] == True]
c_ = len(ratings)-1
n_obligors = db_credit.values[:, :c_+1]
dates = np.array(db_credit.index).astype('datetime64[D]')
t_ = dates.shape[0]
n_cum_trans = db_credit.values[:, c_+1:].reshape(t_, c_+1, c_+1)
stocks_path = '../../../databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(stocks_path + 'db_stocks_sp.csv', skiprows=[0],
                        index_col=0)
v = db_stocks.loc[:, ['GE', 'JPM']].values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step01): Compute the prior transition matrix

# +
p_ = fit_ratings_markov_chain(dates, n_obligors, n_cum_trans, tau_hl)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step02): Compute final transition matrix

# +
# probability constraint
a_eq = np.ones((1, c_+1))
b_eq = np.array([1])
# monotonicity constraint (initialize)
a_ineq = {}
a_ineq[0] = np.diagflat(np.ones((1, c_)), 1) -\
                   np.diagflat(np.ones((1, c_+1)), 0)
a_ineq[0] = a_ineq[0][:-1]
b_ineq = np.zeros((c_))
# relative entropy minimization
p = np.zeros((c_, c_+1))
for c in range(c_):
    p[c, :] = min_rel_entropy_sp(p_[c, :],
                                 a_ineq[c], b_ineq, a_eq, b_eq, False)
    # update monotonicity constraint
    a_temp = a_ineq.get(c).copy()
    a_temp[c, :] = -a_temp[c, :]
    a_ineq[c+1] = a_temp.copy()

# default constraint
p = np.r_[p, np.array([np.r_[np.zeros(7), 1]])]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step03): Compute cdf

# +
f = np.cumsum(p[r-1, :])
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step04): Save database

# +
out = pd.DataFrame(p)
out.to_csv(path+'db_trans_matrix.csv')
del out
# -

# ## Plots

# +
fig = plt.figure()
plt.style.use('arpm')
plt.step(np.arange(c_+2), np.r_[0, f])
add_logo(fig)
# -


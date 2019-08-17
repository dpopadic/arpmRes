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

# # s_checklist_scenariobased_step02 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step02&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-2).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation import fit_garch_fp, fit_ratings_markov_chain, \
                             fit_var1
from arpym.statistics import invariance_test_copula, \
                             invariance_test_ellipsoid, \
                             invariance_test_ks
from arpym.views import min_rel_entropy_sp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-parameters)

tau_hl_credit = 5  # half-life parameter for credit fit (years)
i_plot = 1  # select the invariant to be tested (i = 1,...,i_)
lag_ = 5  # lag used in invariance tests

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

# market risk drivers
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series.csv',
                                    index_col=0, parse_dates=True)
x = db_riskdrivers_series.values
dates = pd.to_datetime(np.array(db_riskdrivers_series.index))
risk_drivers_names = db_riskdrivers_series.columns.values

# additional information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
d_implvol = int(db_riskdrivers_tools.d_implvol.dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna())
tlast_credit = np.datetime64(db_riskdrivers_tools.tlast_credit[0], 'D')
c_ = int(db_riskdrivers_tools.c_.dropna())
ratings_param = db_riskdrivers_tools.ratings_param.dropna()

i_bonds = n_bonds*4  # 4 NS parameters x n_bonds
ind_ns_bonds = np.arange(n_stocks+1+d_implvol,
                         n_stocks+1+d_implvol+i_bonds)

# credit risk drivers
db_riskdrivers_credit = pd.read_csv(path+'db_riskdrivers_credit.csv',
                                    index_col=0, parse_dates=True)
dates_credit = np.array(db_riskdrivers_credit.index).astype('datetime64[D]')

# number of obligors
n_obligors = db_riskdrivers_credit.iloc[:, :c_+1]

# cumulative number of transitions
n_cum_trans = db_riskdrivers_credit.iloc[:, c_+1:(c_+1)**2]
from_to_index = pd.MultiIndex.from_product([ratings_param, ratings_param],
                                           names=('rating_from', 'rating_to'))
mapper = {}
for col in n_cum_trans:
    (rating_from, _, rating_to) = col[12:].partition('_')
    mapper[col] = (rating_from, rating_to)
n_cum_trans = n_cum_trans.rename(columns=mapper) \
                                     .reindex(columns=from_to_index).fillna(0)

del db_riskdrivers_tools, db_riskdrivers_credit

t_ = len(dates)-1  # length of the invariants time series

# initialize temporary databases
db_invariants = {}
db_nextstep = {}
db_garch_sig2 = {}
db_param = {}
param_names = ['a', 'b', 'c', 'mu']
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step01): GARCH(1,1) fit on stocks log-values

for i in range(n_stocks):
    # time series of risk driver increment
    dx = np.diff(x[:, i])
    # fit parameters
    par, sig2, epsi, *_ = fit_garch_fp(dx)
    # store next-step function and invariants
    db_invariants[i] = np.array(epsi)
    db_param[i] = par
    db_nextstep[i] = 'GARCH(1,1)'
    db_garch_sig2[i] = sig2

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step02): GARCH(1,1) fit on S&P index log-values

# time series of risk driver increment
dx = np.diff(x[:, n_stocks])
# fit parameters
par, sig2, epsi, *_ = fit_garch_fp(dx)
# store next-step function and invariants
db_invariants[n_stocks] = np.array(epsi)
db_param[n_stocks] = par
db_nextstep[n_stocks] = 'GARCH(1,1)'
db_garch_sig2[n_stocks] = sig2

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step03): Random walk fit on options log-implied volatility

for i in range(n_stocks+1, n_stocks+1+d_implvol):
    db_invariants[i] = np.diff(x[:, i])
    db_nextstep[i] = 'Random walk'
    db_param[i] = np.full(4, np.nan)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step04): AR(1) fit of Nelson-Siegel parameters

# +
# the fit is performed only on non-nan entries
t_bonds = np.sum(np.isfinite(x[:, ind_ns_bonds[0]]))

x_obligor = np.zeros((t_bonds, i_bonds))
epsi_obligor = np.zeros((t_bonds-1, i_bonds))

b_ar_obligor = np.zeros(i_bonds)  # initialize AR(1) parameter
for i in range(i_bonds):
    # risk driver (non-nan entries)
    x_obligor[:, i] = x[t_-t_bonds+1:, ind_ns_bonds[i]]
    # fit parameter
    b_ar_obligor[i], _, _ = fit_var1(x_obligor[:, i])
    # invariants
    epsi_obligor[:, i] = x_obligor[1:, i]-b_ar_obligor[i]*x_obligor[:-1, i]

# store the next-step function and the extracted invariants
k = 0
for i in ind_ns_bonds:
    db_invariants[i] = np.r_[np.full(t_-t_bonds+1, np.nan),
                             epsi_obligor[:, k]]
    db_nextstep[i] = 'AR(1)'
    tmp = np.full(4, np.nan)
    tmp[1] = b_ar_obligor[k]
    db_param[i] = tmp
    k = k+1
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step05): Credit migrations: time-homogeneous Markov chain

# +
# array format
n_cum_trans = n_cum_trans.values.reshape((-1, c_+1, c_+1), order='C')

# estimation of annual credit transition matrix
# compute the prior credit transition matrix via maximum likelihood
p_credit_prior = fit_ratings_markov_chain(dates_credit,
                                          n_obligors.values,
                                          n_cum_trans, tau_hl_credit)

# constraints on the credit transition matrix via minimum relative entropy
# probability constraint
a = {}
a[0] = np.diagflat(np.ones((1, c_)), 1) -\
                   np.diagflat(np.ones((1, c_+1)), 0)
a[0] = a[0][:-1]
b = np.zeros((c_))
# monotonicity constraint (initialize)
a_eq = np.ones((1, c_+1))
b_eq = np.array([1])
# minimum relative entropy
p_credit = np.zeros((c_, c_+1))
for c in range(c_):
    p_credit[c, :] = min_rel_entropy_sp(p_credit_prior[c, :],
                                        a[c], b, a_eq, b_eq, False)
    # update monotonicity constraint
    a_temp = a.get(c).copy()
    a_temp[c, :] = -a_temp[c, :]
    a[c+1] = a_temp.copy()

# default constraint
default_constraint = np.append(np.zeros(c_), 1).reshape(1, -1)
p_credit = np.r_[p_credit, default_constraint]
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step06): Save databases

# +
dates = dates[1:]

# all market invariants
out = pd.DataFrame({risk_drivers_names[i]: db_invariants[i]
                    for i in range(len(db_invariants))}, index=dates)
out = out[list(risk_drivers_names[:len(db_invariants)])]
out.index.name = 'dates'
out.to_csv(path+'db_invariants_series.csv')
del out

# next-step models for all invariants
out = pd.DataFrame({risk_drivers_names[i]: db_nextstep[i]
                    for i in range(len(db_nextstep))}, index=[''])
out = out[list(risk_drivers_names[:len(db_nextstep)])]
out.to_csv(path+'db_invariants_nextstep.csv',
           index=False)
del out

# all parameters
out = pd.DataFrame({risk_drivers_names[i]: db_param[i]
                    for i in range(len(db_param))}, index=param_names)
out = out[list(risk_drivers_names[:len(db_param)])]
out.to_csv(path+'db_invariants_param.csv')
del out

# squared dispersion parameters in GARCH(1,1) model
out = pd.DataFrame({risk_drivers_names[i]: db_garch_sig2[i]
                    for i in range(len(db_garch_sig2))}, index=dates)
out.index.name = 'dates'
out = out[list(risk_drivers_names[:len(db_garch_sig2)])]
out.to_csv(path+'db_garch_sig2.csv')
del out

# annual credit transition matrix
out = pd.DataFrame({'p_credit': pd.Series(p_credit.reshape(-1))})
out.to_csv(path+'db_invariants_p_credit.csv',
           index=None)
del out
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step02-implementation-step07): Perform invariance tests

# +
plt.style.use('arpm')

invar = db_invariants[i_plot-1][~np.isnan(db_invariants[i_plot-1])]

_ = invariance_test_ellipsoid(invar, lag_)
fig_ellipsoid = plt.gcf()
fig_ellipsoid.set_dpi(72.0)
fig_ellipsoid.set_size_inches(1280.0/72.0, 720.0/72.0)
add_logo(fig_ellipsoid, set_fig_size=False)
plt.show()

invariance_test_ks(invar)
fig_ks = plt.gcf()
fig_ks.set_dpi(72.0)
fig_ks.set_size_inches(1280.0/72.0, 720.0/72.0)
add_logo(fig_ks, set_fig_size=False)
plt.tight_layout()

_ = invariance_test_copula(invar, lag_)
fig_cop = plt.gcf()
fig_cop.set_dpi(72.0)
fig_cop.set_size_inches(1280.0/72.0, 720.0/72.0)
plt.tight_layout()

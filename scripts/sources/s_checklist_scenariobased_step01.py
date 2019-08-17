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

# # s_checklist_scenariobased_step01 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step01&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-1).

# +
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.pricing import bsm_function, bootstrap_nelson_siegel, \
                          implvol_delta2m_moneyness
from arpym.tools import aggregate_rating_migrations, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-parameters)

# +
# set current time t_now
t_now = np.datetime64('2012-08-31')

# set start date for data selection
t_first = np.datetime64('2009-11-02')

# set initial portfolio construction date t_init
t_init = np.datetime64('2012-08-30')

# stocks - must include GE and JPM
stock_names = ['GE', 'JPM', 'A', 'AA', 'AAPL']  # stocks considered
# make sure stock names includes GE and JPM
stock_names = ['GE', 'JPM'] + [stock
                               for stock in stock_names
                               if stock not in ['GE', 'JPM']]
print('Stocks considered:', stock_names)

# options on S&P 500
k_strk = 1407  # strike value of options on S&P 500 (US dollars)
tend_option = np.datetime64('2013-08-26')  # options expiry date
y = 0.01  # level for yield curve (assumed flat and constant)
l_ = 9  # number of points on the m-moneyness grid

# corporate bonds
# expiry date of the GE coupon bond to extract
tend_ge = np.datetime64('2013-09-16')
# expiry date of the JPM coupon bond to extract
tend_jpm = np.datetime64('2014-01-15')

# starting ratings following the table:
# "AAA" (0), "AA" (1), "A" (2), "BBB" (3), "BB" (4), "B" (5),
# "CCC" (6), "D" (7)
ratings_tnow = np.array([5,   # initial credit rating for GE (corresponding to B)
                         3])  # initial credit rating for JPM  (corresponding to BBB)

# start of period for aggregate credit risk drivers
tfirst_credit = np.datetime64('1995-01-01')
# end of period for aggregate credit risk drivers
tlast_credit = np.datetime64('2004-12-31')

# index of risk driver to plot
d_plot = 1
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step00): Import data

# +
# upload data
# stocks
stocks_path = '../../../databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(stocks_path + 'db_stocks_sp.csv', skiprows=[0],
                        index_col=0)
db_stocks.index = pd.to_datetime(db_stocks.index)

# implied volatility of option on S&P 500 index
path = '../../../databases/global-databases/derivatives/db_implvol_optionSPX/'
db_impliedvol = pd.read_csv(path + 'data.csv',
                            index_col=['date'], parse_dates=['date'])
implvol_param = pd.read_csv(path + 'params.csv', index_col=False)

# corporate bonds: GE and JPM
jpm_path = \
    '../../../databases/global-databases/fixed-income/db_corporatebonds/JPM/'
db_jpm = pd.read_csv(jpm_path + 'data.csv',
                     index_col=['date'], parse_dates=['date'])
jpm_param = pd.read_csv(jpm_path + 'params.csv',
                        index_col=['expiry_date'], parse_dates=['expiry_date'])
jpm_param['link'] = ['dprice_'+str(i) for i in range(1, jpm_param.shape[0]+1)]

ge_path = '../../../databases/global-databases/fixed-income/db_corporatebonds/GE/'
db_ge = pd.read_csv(ge_path + 'data.csv',
                    index_col=['date'], parse_dates=['date'])
ge_param = pd.read_csv(ge_path + 'params.csv',
                       index_col=['expiry_date'], parse_dates=['expiry_date'])
ge_param['link'] = ['dprice_'+str(i) for i in range(1, ge_param.shape[0]+1)]

# ratings
rating_path = '../../../databases/global-databases/credit/db_ratings/'
db_ratings = pd.read_csv(rating_path+'data.csv', parse_dates=['date'])
# ratings_param represents all possible ratings i.e. AAA, AA, etc.
ratings_param = pd.read_csv(rating_path+'params.csv', index_col=0)
ratings_param = np.array(ratings_param.index)
c_ = len(ratings_param)-1

# define the date range of interest
dates = db_stocks.index[(db_stocks.index >= t_first) &
                        (db_stocks.index <= t_now)]
dates = np.intersect1d(dates, db_impliedvol.index)
dates = dates.astype('datetime64[D]')

# the corporate bonds time series is shorter; select the bonds dates
ind_dates_bonds = np.where((db_ge.index >= dates[0]) &
                           (db_ge.index <= t_now))
dates_bonds = np.intersect1d(db_ge.index[ind_dates_bonds], db_jpm.index)
dates_bonds = dates_bonds.astype('datetime64[D]')

# length of the time series
t_ = len(dates)
t_bonds = len(dates_bonds)

# initialize temporary databases
db_risk_drivers = {}
v_tnow = {}
v_tinit = {}
risk_drivers_names = {}
v_tnow_names = {}

# implied volatility parametrized by time to expiry and delta-moneyness
tau_implvol = np.array(implvol_param.time2expiry)
tau_implvol = tau_implvol[~np.isnan(tau_implvol)]
delta_moneyness = np.array(implvol_param.delta)

implvol_delta_moneyness_2d = \
    db_impliedvol.loc[(db_impliedvol.index.isin(dates)),
                      (db_impliedvol.columns != 'underlying')]

k_ = len(tau_implvol)

# unpack flattened database (from 2d to 3d)
implvol_delta_moneyness_3d = np.zeros((t_, k_, len(delta_moneyness)))
for k in range(k_):
    implvol_delta_moneyness_3d[:, k, :] = \
        np.r_[np.array(implvol_delta_moneyness_2d.iloc[:, k::k_])]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step01): Stocks

# +
n_stocks = len(stock_names)  # number of stocks
d_stocks = n_stocks  # one risk driver for each stock

for d in range(d_stocks):
    # calculate time series of stock risk drivers
    db_risk_drivers[d] = np.log(np.array(db_stocks.loc[dates, stock_names[d]]))
    risk_drivers_names[d] = 'stock '+stock_names[d]+'_log_value'
    # stock value
    v_tnow[d] = db_stocks.loc[t_now, stock_names[d]]
    v_tinit[d] = db_stocks.loc[t_init, stock_names[d]]
    v_tnow_names[d] = 'stock '+stock_names[d]

# number of risk drivers, to be updated at every insertion
d_ = d_stocks
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step02): S&P 500 Index

# +
# calculate risk driver of the S&P 500 index
db_risk_drivers[d_] = \
    np.log(np.array(db_impliedvol.loc[(db_impliedvol.index.isin(dates)),
                                      'underlying']))
risk_drivers_names[d_] = 'sp_index_log_value'

# value of the S&P 500 index
v_tnow[d_] = db_impliedvol.loc[t_now, 'underlying']
v_tinit[d_] = db_impliedvol.loc[t_init, 'underlying']
v_tnow_names[d_] = 'sp_index'

# update counter
d_ = d_+1
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step03): Call and put options on the S&P 500 Index

# +
# from delta-moneyness to m-moneyness parametrization
implvol_m_moneyness_3d, m_moneyness = \
    implvol_delta2m_moneyness(implvol_delta_moneyness_3d, tau_implvol,
                              delta_moneyness, y*np.ones((t_, k_)),
                              tau_implvol, l_)

# calculate log implied volatility
log_implvol_m_moneyness_2d = \
    np.log(np.reshape(implvol_m_moneyness_3d,
                      (t_, k_*(l_)), 'F'))

# value of the underlying
s_tnow = v_tnow[d_stocks]
s_tinit = v_tinit[d_stocks]

# time to expiry (in years)
tau_option_tnow = np.busday_count(t_now, tend_option)/252
tau_option_tinit = np.busday_count(t_init, tend_option)/252

# moneyness
moneyness_tnow = np.log(s_tnow/k_strk)/np.sqrt(tau_option_tnow)
moneyness_tinit = np.log(s_tnow/k_strk)/np.sqrt(tau_option_tnow)

# grid points
points = list(zip(*[grid.flatten() for grid in np.meshgrid(*[tau_implvol,
                                                             m_moneyness])]))

# known values
values = implvol_m_moneyness_3d[-1, :, :].flatten('F')

# implied volatility (interpolated)
impl_vol_tnow = \
    interpolate.LinearNDInterpolator(points, values)(*np.r_[tau_option_tnow,
                                                            moneyness_tnow])
impl_vol_tinit = \
    interpolate.LinearNDInterpolator(points, values)(*np.r_[tau_option_tinit,
                                                            moneyness_tinit])

# compute call option value by means of Black-Scholes-Merton formula
v_call_tnow = bsm_function(s_tnow, y, impl_vol_tnow, moneyness_tnow, tau_option_tnow)
v_call_tinit = bsm_function(s_tinit, y, impl_vol_tinit, moneyness_tinit,
                            tau_option_tinit)

# compute put option value by means of the put-call parity
v_zcb_tnow = np.exp(-y*tau_option_tnow)
v_put_tnow = v_call_tnow - s_tnow + k_strk*v_zcb_tnow
v_zcb_tinit = np.exp(-y*tau_option_tinit)
v_put_tinit = v_call_tinit - s_tinit + k_strk*v_zcb_tinit

# store data
d_implvol = log_implvol_m_moneyness_2d.shape[1]
for d in np.arange(d_implvol):
    db_risk_drivers[d_+d] = log_implvol_m_moneyness_2d[:, d]
    risk_drivers_names[d_+d] = 'option_spx_logimplvol_mtau_' + str(d+1)

v_tnow[d_] = v_call_tnow
v_tinit[d_] = v_call_tinit
v_tnow_names[d_] = 'option_spx_call'
v_tnow[d_+1] = v_put_tnow
v_tinit[d_+1] = v_put_tinit
v_tnow_names[d_+1] = 'option_spx_put'

# update counter
d_ = len(db_risk_drivers)
n_ = len(v_tnow)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step04): Corporate bonds

# +
n_bonds = 2

# GE bond

# extract coupon
coupon_ge = ge_param.loc[tend_ge, 'coupons']/100

# rescaled dirty prices of GE bond
v_bond_ge = db_ge.loc[db_ge.index.isin(dates_bonds)]/100

# computation of Nelson-Siegel parameters for GE bond
theta_ge = np.zeros((t_bonds, 4))
theta_ge = bootstrap_nelson_siegel(v_bond_ge.values, dates_bonds,
                                   np.array(ge_param.coupons/100),
                                   ge_param.index.values.astype('datetime64[D]'))

# risk drivers for bonds are Nelson-Siegel parameters
for d in np.arange(4):
    if d == 3:
        db_risk_drivers[d_+d] = np.sqrt(theta_ge[:, d])
    else:
        db_risk_drivers[d_+d] = theta_ge[:, d]
    risk_drivers_names[d_+d] = 'ge_bond_nel_sieg_theta_' + str(d+1)

# store dirty price of GE bond
# get column variable name in v_bond_ge that selects bond with correct expiry
ge_link = ge_param.loc[tend_ge, 'link']
v_tnow[n_] = v_bond_ge.loc[t_now, ge_link]
v_tinit[n_] = v_bond_ge.loc[t_init, ge_link]
v_tnow_names[n_] = 'ge_bond'

# update counter
d_ = len(db_risk_drivers)
n_ = len(v_tnow_names)

# JPM bond

# extract coupon
coupon_jpm = jpm_param.loc[tend_jpm, 'coupons']/100

# rescaled dirty prices of JPM bond
v_bond_jpm = db_jpm.loc[db_ge.index.isin(dates_bonds)]/100

# computation of Nelson-Siegel parameters for JPM bond
theta_jpm = np.zeros((t_bonds, 4))
theta_jpm = bootstrap_nelson_siegel(v_bond_jpm.values, dates_bonds,
                                   np.array(jpm_param.coupons/100),
                                   jpm_param.index.values.astype('datetime64[D]'))

# risk drivers for bonds are Nelson-Siegel parameters
for d in np.arange(4):
    if d == 3:
        db_risk_drivers[d_+d] = np.sqrt(theta_jpm[:, d])
    else:
        db_risk_drivers[d_+d] = theta_jpm[:, d]
    risk_drivers_names[d_+d] = 'jpm_bond_nel_sieg_theta_'+str(d+1)

# store dirty price of JPM bond
# get column variable name in v_bond_ge that selects bond with correct expiry
jpm_link = jpm_param.loc[tend_jpm, 'link']
v_tnow[n_] = v_bond_jpm.loc[t_now, jpm_link]
v_tinit[n_] = v_bond_jpm.loc[t_init, jpm_link]
v_tnow_names[n_] = 'jpm_bond'

# update counter
d_ = len(db_risk_drivers)
n_ = len(v_tnow)

# fill the missing values with nan's
for d in range(d_stocks+1+d_implvol,
               d_stocks+1+d_implvol+n_bonds*4):
    db_risk_drivers[d] = np.concatenate((np.zeros(t_-t_bonds),
                                         db_risk_drivers[d]))
    db_risk_drivers[d][:t_-t_bonds] = np.NAN
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step05): Credit

# +
# extract aggregate credit risk drivers
dates_credit, n_obligors, n_cum_trans, *_ = \
    aggregate_rating_migrations(db_ratings, ratings_param, tfirst_credit,
                                tlast_credit)

# number of obligors in each rating at each t
t_credit = len(dates_credit)  # length of the time series
credit_types = {}
credit_series = {}
for c in np.arange(c_+1):
    credit_types[c] = 'n_oblig_in_state_'+ratings_param[c]
    credit_series[c] = n_obligors[:, c]

d_credit = len(credit_series)

# cumulative number of migrations up to time t for each pair of rating buckets
for i in np.arange(c_+1):
    for j in np.arange(c_+1):
        if i != j:
            credit_types[d_credit] = \
                'n_cum_trans_'+ratings_param[i]+'_'+ratings_param[j]
            credit_series[d_credit] = n_cum_trans[:, i, j]
            d_credit = len(credit_series)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step01-implementation-step06): Save databases

# +
path = '../../../databases/temporary-databases/'

# market risk drivers
out = pd.DataFrame({risk_drivers_names[d]: db_risk_drivers[d]
                    for d in range(len(db_risk_drivers))}, index=dates)
out = out[list(risk_drivers_names.values())]
out.index.name = 'dates'
out.to_csv(path+'db_riskdrivers_series.csv')
del out

# aggregate credit risk drivers
out = pd.DataFrame({credit_types[d]: credit_series[d]
                    for d in range(d_credit)},
                   index=dates_credit)
out = out[list(credit_types.values())]
out.index.name = 'dates'
out.to_csv(path+'db_riskdrivers_credit.csv')
del out

# values of all instruments at t_now
out = pd.DataFrame({v_tnow_names[n]: pd.Series(v_tnow[n])
                   for n in range(len(v_tnow))})
out = out[list(v_tnow_names.values())]
out.to_csv(path+'db_v_tnow.csv',
           index=False)
del out

# values of all instruments at t_init
out = pd.DataFrame({v_tnow_names[n]: pd.Series(v_tinit[n])
                   for n in range(len(v_tinit))})
out = out[list(v_tnow_names.values())]
out.to_csv(path+'db_v_tinit.csv',
           index=False)
del out

# additional variables needed for subsequent steps
out = {'n_stocks': pd.Series(n_stocks),
       'd_implvol': pd.Series(d_implvol),
       'n_bonds': pd.Series(n_bonds),
       'c_': pd.Series(c_),
       'tlast_credit': pd.Series(tlast_credit),
       'tend_option': pd.Series(tend_option),
       'k_strk': pd.Series(k_strk),
       'l_': pd.Series(l_),
       'tau_implvol': pd.Series(tau_implvol),
       'y': pd.Series(y),
       'm_moneyness': pd.Series(m_moneyness),
       'tend_ge': pd.Series(tend_ge),
       'coupon_ge': pd.Series(coupon_ge),
       'tend_jpm': pd.Series(tend_jpm),
       'coupon_jpm': pd.Series(coupon_jpm),
       'd_': pd.Series(d_),
       'd_credit': pd.Series(d_credit),
       'ratings_tnow': pd.Series(ratings_tnow),
       'ratings_param': pd.Series(ratings_param),
       'stock_names': pd.Series(stock_names),
       't_now': pd.Series(t_now),
       't_init': pd.Series(t_init)}
out = pd.DataFrame(out)
out.to_csv(path+'db_riskdrivers_tools.csv',
           index=False)
del out
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.plot(dates, db_risk_drivers[d_plot-1])
plt.title(risk_drivers_names[d_plot-1], fontweight='bold', fontsize=20)
plt.xlabel('time (days)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([dates[0], dates[-1]])
add_logo(fig, set_fig_size=False)
fig.tight_layout()

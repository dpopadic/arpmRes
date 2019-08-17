# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_checklist_scenariobased_step05 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step05&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-5).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.pricing import bond_value, bsm_function, cash_flow_reinv
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-parameters)

# +
# indicates which projection to continue from
# True: use copula-marginal projections
# False: use historical projections
copula_marginal = True

recrate_ge = 0.6  # recovery rate for GE bond
recrate_jpm = 0.7  # recovery rate for JPM bond
n_plot = 3  # index of instrument to plot
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

# Risk drivers identification
# risk driver values
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series.csv',
                                    index_col=0)
x = db_riskdrivers_series.values

# values at t_now
db_v_tnow = pd.read_csv(path+'db_v_tnow.csv')
v_tnow = db_v_tnow.values[0]

# additional information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv',
                                  parse_dates=True)
d_ = int(db_riskdrivers_tools['d_'].dropna())
n_stocks = int(db_riskdrivers_tools['n_stocks'].dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna())
n_ = n_stocks+n_bonds+3
d_implvol = int(db_riskdrivers_tools['d_implvol'].dropna())
tend_option = np.datetime64(db_riskdrivers_tools['tend_option'][0], 'D')
k_strk = db_riskdrivers_tools['k_strk'][0]
l_ = int(db_riskdrivers_tools['l_'].dropna())
m_moneyness = db_riskdrivers_tools['m_moneyness'].values[:l_]
tau_implvol = db_riskdrivers_tools['tau_implvol'].values
y = db_riskdrivers_tools['y'][0]
tend_ge = np.datetime64(db_riskdrivers_tools['tend_ge'][0], 'D')
tend_jpm = np.datetime64(db_riskdrivers_tools['tend_jpm'][0], 'D')
coupon_ge = db_riskdrivers_tools['coupon_ge'][0]
coupon_jpm = db_riskdrivers_tools['coupon_jpm'][0]
c_ = int(db_riskdrivers_tools.c_.dropna())
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')
# index of risk drivers for options and bonds
idx_options = np.array(range(n_stocks+1, n_stocks+d_implvol+1))
idx_gebond = np.array(range(n_stocks+d_implvol+1, n_stocks+d_implvol+5))
idx_jpmbond = np.array(range(n_stocks+d_implvol+5, n_stocks+d_implvol+9))

# Projection

# load projections from copula-marginal approach
if copula_marginal:
    # projected risk driver paths
    db_projection_riskdrivers = pd.read_csv(path+'db_projection_riskdrivers.csv')

    # projected rating paths
    db_projection_ratings = pd.read_csv(path+'db_projection_ratings.csv')

    # projected scenarios probabilities
    db_scenario_probs = pd.read_csv(path+'db_scenario_probs.csv')
    p = db_scenario_probs['p'].values

    # additional information
    db_projection_tools = pd.read_csv(path+'db_projection_tools.csv')
    j_ = int(db_projection_tools['j_'][0])
    t_hor = np.datetime64(db_projection_tools['t_hor'][0], 'D')
# load projections from historical approach
else:
    # projected risk driver paths
    db_projection_riskdrivers = \
        pd.read_csv(path+'db_projection_bootstrap_riskdrivers.csv')

    # projected scenarios probabilities
    db_scenario_probs = pd.read_csv(path+'db_scenario_probs_bootstrap.csv')
    p = db_scenario_probs['p'].values

    # additional information
    db_projection_tools = \
        pd.read_csv(path+'db_projection_bootstrap_tools.csv')
    j_ = int(db_projection_tools['j_'][0])
    t_hor = np.datetime64(db_projection_tools['t_hor'][0], 'D')
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step01): Calculate number of business days between t_now and t_hor

# +
# business days between t_now and t_hor
m_ = np.busday_count(t_now, t_hor)
# date of next business day (t_now + 1)
t_1 = np.busday_offset(t_now, 1, roll='forward')

# projected scenarios
x_proj = db_projection_riskdrivers.values.reshape(j_, m_+1, d_)
if copula_marginal:
    # projected ratings
    proj_ratings = db_projection_ratings.values.reshape((j_, m_+1, 2))
# initialize output arrays
pi_tnow_thor = np.zeros((j_, n_))
pi_oneday = np.zeros((j_, n_))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step02): Stocks

for n in range(n_stocks):
    pi_tnow_thor[:, n] = v_tnow[n] * (np.exp(x_proj[:, -1, n] - x[-1, n])-1)
    pi_oneday[:, n] = v_tnow[n] * (np.exp(x_proj[:, 1, n] - x[-1, n])-1)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step03): S&P index

pi_tnow_thor[:, n_stocks] = v_tnow[n_stocks]*(np.exp(x_proj[:, -1, n_stocks] -
                                               x[-1, n_stocks])-1)
pi_oneday[:, n_stocks] = v_tnow[n_stocks]*(np.exp(x_proj[:, 1, n_stocks] -
                                               x[-1, n_stocks])-1)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step04): Options

# +
# time to expiry of the options at the horizon t_hor
tau_opt_thor = np.busday_count(t_hor, tend_option)/252
# time to expiry of the options after one day
tau_opt_oneday = np.busday_count(t_1, tend_option)/252

# underlying and moneyness at the horizon
s_thor = np.exp(x_proj[:, -1, n_stocks])
mon_thor = np.log(s_thor/k_strk)/np.sqrt(tau_opt_thor)
# underlying and moneyness after one day
s_oneday = np.exp(x_proj[:, 1, n_stocks])
mon_oneday = np.log(s_oneday/k_strk)/np.sqrt(tau_opt_oneday)

# log-implied volatility at the horizon
logsigma_thor = x_proj[:, -1, idx_options].reshape(j_, -1, l_)
# log-implied volatility after one day
logsigma_oneday = x_proj[:, 1, idx_options].reshape(j_, -1, l_)

# interpolate log-implied volatility
logsigma_interp = np.zeros(j_)
logsigma_interp_oneday = np.zeros(j_)
for j in range(j_):
    # grid points
    points = list(zip(*[grid.flatten()
                        for grid in np.meshgrid(*[tau_implvol, m_moneyness])]))
    # known values
    values = logsigma_thor[j, :, :].flatten('F')
    values_oneday = logsigma_oneday[j, :, :].flatten('F')
    # interpolation
    moneyness_thor = min(max(mon_thor[j], min(m_moneyness)), max(m_moneyness))
    moneyness_oneday = min(max(mon_oneday[j], min(m_moneyness)), max(m_moneyness))
    # log-implied volatility at the horizon
    logsigma_interp[j] =\
        interpolate.LinearNDInterpolator(points, values)(*np.r_[tau_opt_thor,
                                                                moneyness_thor])
    # log-implied volatility after one day
    logsigma_interp_oneday[j] =\
        interpolate.LinearNDInterpolator(points, values_oneday)(*np.r_[tau_opt_oneday,
                                                                       moneyness_oneday])

# compute call option value by means of Black-Scholes-Merton formula
v_call_thor = bsm_function(s_thor, y, np.exp(logsigma_interp), moneyness_thor,
                           tau_opt_thor)
v_call_oneday = bsm_function(s_oneday, y, np.exp(logsigma_interp_oneday), moneyness_oneday,
                           tau_opt_oneday)

# compute put option value using put-call parity
v_zcb_thor = np.exp(-y*tau_opt_thor)
v_put_thor = v_call_thor - s_thor + k_strk*v_zcb_thor
v_zcb_oneday = np.exp(-y*tau_opt_oneday)
v_put_oneday = v_call_oneday - s_oneday + k_strk*v_zcb_oneday

# compute P&L of the call option
pi_tnow_thor[:, n_stocks+1] = v_call_thor - v_tnow[n_stocks+1]
pi_oneday[:, n_stocks+1] = v_call_oneday - v_tnow[n_stocks+1]
# compute P&L of the put option
pi_tnow_thor[:, n_stocks+2] = v_put_thor - v_tnow[n_stocks+2]
pi_oneday[:, n_stocks+2] = v_put_oneday - v_tnow[n_stocks+2]
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step05): Bonds value path without credit risk

# +
# GE

# dates of coupon payments from t_now to time of maturity
# assumed to be equal to record dates
r_ge = np.flip(pd.date_range(start=tend_ge, end=t_now,
                             freq='-180D'))
r_ge = np.busday_offset(np.array(r_ge).astype('datetime64[D]'),
                        0, roll='forward')

# coupon values
coupon_ge_semi = coupon_ge/2
c_ge = coupon_ge_semi*np.ones(len(r_ge))

# bond values without credit risk
v_gebond_thor = np.zeros((j_, m_+1))
v_gebond_thor[:, 0] = v_tnow[n_stocks+3]

# coupon-bond values
for m in range(1, m_+1):
    t_m = np.busday_offset(t_now, m, roll='forward')
    # Nelson-Siegel parameters
    theta_ge = x_proj[:, m, idx_gebond]
    # last element must be squared
    theta_ge[:, 3] = theta_ge[:, 3]**2
    # coupons paid on or after t_m
    r_ge_tm = r_ge[r_ge >= t_m]
    c_ge_tm = c_ge[r_ge >= t_m]
    v_gebond_thor[:, m] = bond_value(t_m, c_ge_tm, r_ge_tm,
                                     'ns', theta_ge)

# JPM

# dates of coupon payments from t_now to time of maturity
# assumed to be equal to record dates
r_jpm = np.flip(pd.date_range(start=tend_jpm, end=t_now,
                              freq='-180D'))
r_jpm = np.busday_offset(np.array(r_jpm).astype('datetime64[D]'),
                        0, roll='forward')

# coupon values
coupon_jpm_semi = coupon_jpm/2
c_jpm = coupon_jpm_semi*np.ones(len(r_jpm))

# bond values without credit risk
v_jpmbond_thor = np.zeros((j_, m_+1))
v_jpmbond_thor[:, 0] = v_tnow[n_stocks+4]

# coupon-bond values
for m in range(1, m_+1):
    t_m = np.busday_offset(t_now, m, roll='forward')
    # Nelson-Siegel parameters
    theta_jpm = x_proj[:, m, idx_jpmbond]
    # last element must be squared
    theta_jpm[:, 3] = theta_jpm[:, 3]**2
    # coupons paid on or after t_m
    r_jpm_tm = r_jpm[r_jpm >= t_m]
    c_jpm_tm = c_jpm[r_jpm >= t_m]
    v_jpmbond_thor[:, m] = bond_value(t_m, c_jpm_tm, r_jpm_tm,
                                      'ns', theta_jpm)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step06): Reinvested cash flow value time series without credit risk

# +
# investment factor
d_tm = 1/252  # one day
invfact = np.exp(y*d_tm)*np.ones((j_, m_))

# GE

# reinvested cash-flow streams
# select coupons and coupon dates in (t_now, t_hor]
# payment dates from t_now to t_hor
r_ge_cf = r_ge[r_ge < np.datetime64(t_hor, 'D')]
# coupon payments
c_ge_cf = np.ones((len(r_ge_cf)))*coupon_ge_semi
# monitoring dates
tnow_thor_ge = np.array(pd.bdate_range(t_now, min(tend_ge, t_hor)))

# scenarios of cumulative cash-flow path
if len(r_ge_cf) > 0:
    cf_ge = cash_flow_reinv(c_ge_cf, r_ge_cf,
                            tnow_thor_ge, invfact)
else:
    cf_ge = np.zeros((j_, m_))

# JPM

# reinvested cash-flow streams
# select coupons and coupon dates in (t_now, t_hor]
# payment dates from t_now to t_hor
r_jpm_cf = r_jpm[r_jpm < np.datetime64(t_hor, 'D')]
# coupon payments
c_jpm_cf = np.ones((len(r_jpm_cf)))*coupon_jpm_semi
# monitoring dates
tnow_thor_jpm = np.array(pd.bdate_range(t_now, min(tend_jpm, t_hor)))

# scenarios of cumulative cash-flow path
if len(r_jpm_cf) > 0:
    cf_jpm = cash_flow_reinv(c_jpm_cf, r_jpm_cf,
                             tnow_thor_jpm, invfact)
else:
    cf_jpm = np.zeros((j_, m_))
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step07): Bonds and cash flow value with credit risk

# compute the value of the coupon-bonds with credit risk
if copula_marginal:
    # define default indicator
    default = np.any(proj_ratings == c_, axis=1, keepdims=True).squeeze()
    # get time of default
    m_d = np.full((j_, 2), 0, dtype='int')
    for n in range(2):
        for j in range(j_):
            if default[j, n]:
                # get m for first date of default
                m_d[j, n] = np.where(proj_ratings[j, :, n]==c_)[0][0]
                # set projected P&L of the underlying stock to 0
                pi_tnow_thor[j, n] = -v_tnow[n]
                if proj_ratings[j, 1, n]==c_:
                    pi_oneday[j, n] = -v_tnow[n]

    # bond value with market and credit risk at t_hor
    v_mc_gebond_thor = v_gebond_thor[:, -1].copy()
    v_mc_jpmbond_thor = v_jpmbond_thor[:, -1].copy()
    # bond value with market and credit risk after one day
    v_mc_gebond_oneday = v_gebond_thor[:, 1].copy()
    v_mc_jpmbond_oneday = v_jpmbond_thor[:, 1].copy()
    # reinvested cash-flow values at t_hor
    cf_mc_ge = cf_ge[:, -1].copy()
    cf_mc_jpm = cf_jpm[:, -1].copy()
    # reinvested cash-flow values after one day
    cf_mc_ge_oneday = cf_ge[:, 0].copy()
    cf_mc_jpm_oneday = cf_jpm[:, 0].copy()

    for j in range(j_):
        # GE
        if default[j, 0]:  # if default occurs
            if m_d[j, 0]==1:  # if default at the first future horizon
                v_mc_gebond_thor[j] = v_tnow[n_stocks+3]*recrate_ge
                cf_mc_ge[j] = 0
                # one day values
                v_mc_gebond_oneday[j] = v_tnow[n_stocks+3]*recrate_ge
                cf_mc_ge_oneday[j] = 0
            else:
                # bond value with credit risk
                v_mc_gebond_thor[j] = \
                    v_gebond_thor[j, int(m_d[j, 0])-1]*recrate_ge
                # cash-flow with credit risk
                cf_mc_ge[j] = cf_ge[j, int(m_d[j, 0])-1]* \
                           np.prod(invfact[j, int(m_d[j, 0]):])
        # JPM
        if default[j, 1]:  # if default occurs
            if m_d[j, 1]==1:  # if default at the first future horizon
                v_mc_jpmbond_thor[j] = v_tnow[n_stocks+4]*recrate_jpm
                cf_mc_jpm[j] = 0
                # one day values
                v_mc_jpmbond_oneday[j] = v_tnow[n_stocks+4]*recrate_jpm
                cf_mc_jpm_oneday[j] = 0
            else:
                # bond value with credit risk
                v_mc_jpmbond_thor[j] = \
                    v_jpmbond_thor[j, int(m_d[j, 1])-1]*recrate_jpm
                # cash-flow with credit risk
                cf_mc_jpm[j] = cf_jpm[j, int(m_d[j, 1])-1]* \
                           np.prod(invfact[j, int(m_d[j, 1]):])

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step08): Bonds projected P&L

# including credit risk
if copula_marginal:
    # compute the ex-ante P&L of bond over [t_now, t_hor)
    pi_tnow_thor[:, n_stocks+3] = v_mc_gebond_thor - \
                                  np.tile(v_tnow[n_stocks+3], j_) + \
                                  cf_mc_ge
    pi_tnow_thor[:, n_stocks+4] = v_mc_jpmbond_thor - \
                                  np.tile(v_tnow[n_stocks+4], j_) + \
                                  cf_mc_jpm
    # compute the ex-ante P&L of bond over one day
    pi_oneday[:, n_stocks+3] = v_mc_gebond_oneday - \
                                  np.tile(v_tnow[n_stocks+3], j_) + \
                                  cf_mc_ge_oneday
    pi_oneday[:, n_stocks+4] = v_mc_jpmbond_oneday - \
                                  np.tile(v_tnow[n_stocks+4], j_) + \
                                  cf_mc_jpm_oneday
# not including credit risk
else:
    # compute the ex-ante P&L of bond over [t_now, t_hor)
    pi_tnow_thor[:, n_stocks+3] = v_gebond_thor[:, m_] - \
                                  np.tile(v_tnow[n_stocks+3], j_) + \
                                  cf_ge[:, m_-1]
    pi_tnow_thor[:, n_stocks+4] = v_jpmbond_thor[:, m_] - \
                                  np.tile(v_tnow[n_stocks+4], j_) + \
                                  cf_jpm[:, m_-1]
    # compute the ex-ante P&L of bond over one day
    pi_oneday[:, n_stocks+3] = v_gebond_thor[:, 1] - \
                                  np.tile(v_tnow[n_stocks+3], j_) + \
                                  cf_ge[:, 0]
    pi_oneday[:, n_stocks+4] = v_jpmbond_thor[:, 1] - \
                                  np.tile(v_tnow[n_stocks+4], j_) + \
                                  cf_jpm[:, 0]


# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step05-implementation-step09): Save database

# +
# ex-ante performance over [t_now, t_hor)
out = {db_v_tnow.columns[n]: pi_tnow_thor[:, n]
       for n in range(n_)}
names = [db_v_tnow.columns[n] for n in range(n_)]
out = pd.DataFrame(out)
out = out[list(names)]
if copula_marginal:
    out.to_csv(path+'db_pricing.csv', index=False)
else:
    out.to_csv(path+'db_pricing_historical.csv',
                index=False)
del out

# ex-ante performance over one day
out = {db_v_tnow.columns[n]: pi_oneday[:, n]
       for n in range(n_)}
names = [db_v_tnow.columns[n] for n in range(n_)]
out = pd.DataFrame(out)
out = out[list(names)]
if copula_marginal:
    out.to_csv(path+'db_oneday_pl.csv', index=False)
else:
    out.to_csv(path+'db_oneday_pl_historical.csv', index=False)
del out
# -

# ## Plots

# +
plt.style.use('arpm')
# instruments P&L plot
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
f, xp = histogram_sp(pi_tnow_thor[:, n_plot-1], p=p, k_=30)

plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.3, .3, .3], edgecolor='k')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('P&L', fontsize=17)
plt.title('Ex-ante P&L: '+db_v_tnow.columns[n_plot-1], fontsize=20, fontweight='bold')

add_logo(fig, set_fig_size=False)

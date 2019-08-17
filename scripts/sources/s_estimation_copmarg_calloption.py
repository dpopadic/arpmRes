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

# # s_estimation_copmarg_calloption [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_estimation_copmarg_calloption&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCopulaEstim).

# +
import numpy as np
import pandas as pd
from scipy.stats import t as tstu
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from arpym.statistics import cop_marg_sep, mvt_pdf, scoring, smoothing
from arpym.estimation import conditional_fp, cov_2_corr, effective_num_scenarios, \
    exp_decay_fp, factor_analysis_paf, \
    fit_garch_fp, fit_locdisp_mlfp
from arpym.tools import add_logo, histogram_sp

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_calloption-parameters)

tau_hl_prior = 4 * 252  # half-life parameter for time conditioning
tau_hl_smooth = 21  # half-life parameter for VIX smoothing
tau_hl_score = 5 * 21  # half-life parameter for VIX scoring
alpha = 0.5  # proportion of obs. included in range for state conditioning
nu_min = 2  # lower bound for the degrees of freedom for t copula
nu_max = 20  # upper bound for the degrees of freedom for t copula
i_plot = 1  # invariant chosed for the plot

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_calloption-implementation-step00): Upload data

# +
# VIX (used for time-state conditioning)
vix_path = '../../../databases/global-databases/derivatives/db_vix/data.csv'
db_vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                     index_col=0)
db_vix.index = pd.to_datetime(db_vix.index)

# S&P 500 index
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
db_sp500 = pd.read_csv(path + 'SPX.csv', index_col=0, parse_dates=True)

path = '../../../databases/temporary-databases/'

# implied volatility (used for dates)
db_calloption_rd = pd.read_csv(path + 'db_calloption_rd.csv', index_col=0,
                               parse_dates=True)
dates = pd.to_datetime(np.array(db_calloption_rd.index))

# invariants extracted from the log-implied volatility
db_calloption_epsi_var1 = pd.read_csv(path + 'db_calloption_epsi_var1.csv',
                                      index_col=0, parse_dates=True)
epsi_var1 = db_calloption_epsi_var1.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_calloption-implementation-step01): Extract invariants for the S&P 500 index and create the realized information panel

# +
# compute risk driver for the S&P 500 index as the log-value
log_underlying = \
    np.log(np.array(db_sp500.loc[(db_sp500.index.isin(dates)), 'SPX_close']))

# model log_underlying as GARCH(1,1)
par, sig2, epsi_garch = fit_garch_fp(np.diff(log_underlying))

# store all the invariants in the realized information panel
epsi = np.c_[epsi_garch, epsi_var1]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_calloption-implementation-step02): Set the flexible probabilities

# +
t_, i_ = epsi.shape
# state indicator: VIX compounded return realizations
c_vix = np.diff(np.log(np.array(db_vix.loc[dates].VIX_close)))
# smoothing
z_smooth = smoothing(c_vix, tau_hl_smooth)
# scoring
z = scoring(z_smooth, tau_hl_score)
# target value
z_star = z[-1]
# prior probabilities
p_prior = exp_decay_fp(t_, tau_hl_prior)
# posterior probabilities
p = conditional_fp(z, z_star, alpha, p_prior)
# effective number of scenarios
ens = effective_num_scenarios(p)

print("Effective number of scenarios is ", int(round(ens)))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_calloption-implementation-step03): Static elliptical copula estimation

# +
# compute the invariants grades
u, _, _ = cop_marg_sep(epsi, p)

# grid for the degrees of freedom parameter
nu_copula = np.arange(nu_min, nu_max + 1)
l_ = len(nu_copula)

# initialize variables
rho2_copula_vec = np.zeros((i_, i_, l_))
llike_nu = np.zeros(l_)
epsi_tilde = np.zeros((t_, i_, l_))

for l in range(l_):
    # t-distributed invariants
    epsi_tilde[:, :, l] = tstu.ppf(u, nu_copula[l])

    # maximum likelihood
    _, sig2 = \
        fit_locdisp_mlfp(epsi_tilde[:, :, l], p=p, nu=nu_copula[l],
                         threshold=10 ** -3, maxiter=1000)

    # compute correlation matrix
    rho2_copula_vec[:, :, l], _ = cov_2_corr(sig2)

    # compute log-likelihood at times with no missing values
    llike_nu[l] = np.sum(p * np.log(mvt_pdf(epsi, np.zeros(i_),
                                            rho2_copula_vec[:, :, l],
                                            nu_copula[l])))
# choose nu that gives the highest log-likelihood
l_max = np.argmax(llike_nu)
nu = nu_copula[l_max]
rho2 = rho2_copula_vec[:, :, l_max]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_calloption-implementation-step04): Save databases

# +
# GARCH(1,1) realized invariants
out = pd.DataFrame({'epsi_log_underlying': epsi_garch}, index=dates[1:])

out.index.name = 'dates'
out.to_csv('../../../databases/temporary-databases/db_calloption_epsi_garch.csv')
del out

# GARCH(1,1) model parameters
out = pd.DataFrame({'a': pd.Series(par[0]),
                    'b': pd.Series(par[1]),
                    'c': pd.Series(par[2]),
                    'mu': pd.Series(par[3]),
                    'sig2prev': pd.Series(sig2[-1]),
                    'x_tnow': pd.Series(log_underlying[-1]),
                    'x_tnow-1': pd.Series(log_underlying[-2])})
out.to_csv('../../../databases/temporary-databases/db_calloption_garch.csv')
del out

# flexible probabilities, copula degrees of freedom and correlation matrix
out = pd.DataFrame({'p': pd.Series(p),
                    'rho2_' + str(0): pd.Series(rho2[0, :])})
for i in range(1, i_):
    out = out.join(pd.DataFrame({'rho2_' + str(i): pd.Series(rho2[:, i])}))
out = out.join(pd.DataFrame({'nu': pd.Series(nu)}))
out.to_csv('../../../databases/temporary-databases/db_calloption_estimation.csv',
           index=None)
del out
# -

# ## Plots

# +
plt.style.use('arpm')

# marginal distribution
fig = plt.figure(figsize=(1280 / 72, 720 / 72), dpi=72)

f_eps, x_eps = histogram_sp(epsi[:, i_plot - 1], p=p, k_=10 * np.log(t_))
bar_width = x_eps[1] - x_eps[0]
plt.bar(x_eps, f_eps.flatten(), width=bar_width, fc=[0.7, 0.7, 0.7],
        edgecolor=[0.5, 0.5, 0.5])

plt.title('Distribution of the selected invariant',
          fontweight='bold', fontsize=20)
plt.xlabel('Invariant', fontsize=17)
add_logo(fig, location=1, set_fig_size=False)
fig.tight_layout()

# copula correlation matrix
fig2 = plt.figure(figsize=(1280.0 / 72.0, 720.0 / 72.0), dpi=72.0)
plt.imshow(rho2_copula_vec[:, :, l_max])
plt.colorbar()
plt.grid(False)
plt.title('Estimated correlation matrix', fontweight='bold', fontsize=20)
add_logo(fig2, size_frac_x=0.8, location=9, alpha=0.8, set_fig_size=False)
fig2.tight_layout()
# -



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

# # s_estimation_copmarg_ratings [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_estimation_copmarg_ratings&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings).

# ## Prepare the environment

# +
import numpy as np
import pandas as pd
from scipy.stats import t as tstu

from arpym.statistics import cop_marg_sep, scoring, smoothing, mvt_pdf
from arpym.estimation import conditional_fp, cov_2_corr, exp_decay_fp, fit_locdisp_mlfp, fit_garch_fp

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings-parameters)

tau_hl_prior = 4 * 252  # half-life parameter for time conditioning
tau_hl_smooth = 21  # half-life parameter for VIX smoothing
tau_hl_score = 5 * 21  # half-life parameter for VIX scoring
alpha = 0.5  # proportion of obs. included in range for state conditioning
nu_min = 2  # lower bound for the degrees of freedom for t copula
nu_max = 20  # upper bound for the degrees of freedom for t copula

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0],
                        index_col=0)
v = db_stocks.loc[:, ['GE', 'JPM']].values

# VIX (used for time-state conditioning)
vix_path = '../../../databases/global-databases/derivatives/db_vix/data.csv'
db_vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                     index_col=0)
db_vix.index = pd.to_datetime(db_vix.index)
dates = pd.to_datetime(db_stocks.loc[::20, ['GE', 'JPM']].index)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings-implementation-step01): Fit GARCH process and extract realized invariants

# select monthly values
v = v[::20, :]
# compute monthly compounded returns
c = np.diff(np.log(v), axis=0)
_, _, epsi_garch_ge = fit_garch_fp(c[:, 0])
_, _, epsi_garch_jpm = fit_garch_fp(c[:, 1])
epsi = np.c_[epsi_garch_ge, epsi_garch_jpm]
t_ = v.shape[0] - 1

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings-implementation-step02): Set the flexible probabilities

# state indicator: VIX compounded return realizations
c_vix = np.diff(np.log(np.array(db_vix.loc[dates, :].VIX_close)))
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

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings-implementation-step03): Estimate t copula

# +
# calculate grades of the compounded returns
u, _, _ = cop_marg_sep(epsi, p)

# grid for the degrees of freedom parameter
nu_copula = np.arange(nu_min, nu_max + 1)
l_ = len(nu_copula)

rho2_copula_vec = np.zeros((2, 2, l_))
llike_nu = np.zeros(l_)

for l in range(l_):
    # t-distributed invariants
    epsi_tilde = tstu.ppf(u, nu_copula[l])

    # maximum likelihood
    _, sig2_hat = fit_locdisp_mlfp(epsi_tilde, nu=nu_copula[l],
                                   threshold=10 ** -3, maxiter=1000)
    # compute correlation matrix
    rho2_copula_vec[:, :, l], _ = cov_2_corr(sig2_hat)

    # compute log-likelihood at times with no missing values
    llike_nu[l] = np.sum(p * np.log(mvt_pdf(epsi, np.zeros(2),
                                            rho2_copula_vec[:, :, l],
                                            nu_copula[l])))

# choose nu that gives the highest log-likelihood
l_max = np.argmax(llike_nu)
nu_hat = nu_copula[l_max]
rho2_hat = rho2_copula_vec[:, :, l_max]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_estimation_copmarg_ratings-implementation-step04): Save database

out = {'rho2': pd.Series(rho2_hat[0, 1]),
       'nu': pd.Series(nu_hat)}
out = pd.DataFrame(out)
path = '../../../databases/temporary-databases/'
out.to_csv(path + 'db_copula_ratings.csv')
del out

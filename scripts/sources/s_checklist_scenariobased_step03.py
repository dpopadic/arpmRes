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

# # s_checklist_scenariobased_step03 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step03&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-3).

# +
import numpy as np
import pandas as pd
from scipy.stats import t as tstu
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from arpym.estimation import conditional_fp, cov_2_corr, effective_num_scenarios, \
    exp_decay_fp, factor_analysis_paf, \
    fit_locdisp_mlfp, fit_locdisp_mlfp_difflength
from arpym.statistics import cop_marg_sep, mvt_pdf, scoring, smoothing, \
    twist_prob_mom_match
from arpym.tools import colormap_fp, histogram_sp, add_logo

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-parameters)

# +
# flexible probabilities parameters
tau_hl_prior = 4 * 252  # half-life parameter for time conditioning (days)
tau_hl_smooth = 21  # half-life parameter for VIX smoothing (days)
tau_hl_score = 5 * 21  # half-life parameter for VIX scoring (days)
alpha = 0.7  # proportion of obs. included in range for state conditioning

# parameters for estimating marginal t distribution
nu_min = 3  # lower bound for the degrees of freedom for t marginals
nu_max = 100  # upper bound for the degrees of freedom for t marginals

# parameters for estimating t copula
nu_min_copula = 3  # lower bound for the degrees of freedom for t copula
nu_max_copula = 5  # upper bound for the degrees of freedom for t copula

# factor analysis
k_ = 10  # number of factors for factor analysis

# modeled invariant to plot
i_plot = 3
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

# VIX (used for time-state conditioning)
vix_path = '../../../databases/global-databases/derivatives/db_vix/data.csv'
db_vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                     index_col=0, parse_dates=True)

# Risk drivers identification
db_riskdrivers_tools = pd.read_csv(path + 'db_riskdrivers_tools.csv')
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna())
d_implvol = int(db_riskdrivers_tools.d_implvol.dropna())
i_bonds = n_bonds * 4  # 4 Nelson-Siegel parameters x n_bonds

# Quest for invariance
db_invariants_series = pd.read_csv(path + 'db_invariants_series.csv',
                                   index_col=0, parse_dates=True)
epsi = db_invariants_series.values
t_, i_ = np.shape(epsi)
dates = np.array(db_invariants_series.index)

db_invariants_nextstep = pd.read_csv(path + 'db_invariants_nextstep.csv')
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step01): Set the flexible probabilities

# +
# time and state conditioning on smoothed and scored VIX returns

# state indicator: VIX compounded return realizations
db_vix['c_vix'] = np.log(db_vix).diff()
# extract data for analysis dates
c_vix = db_vix.c_vix[dates].values
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

print('Effective number of scenarios is', int(round(ens)))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step02): Estimate the marginal distributions for stocks, S&P 500 and implied volatility

# +
# invariants to be modeled parametrically
ind_parametric = np.arange(n_stocks + 1 + d_implvol,
                           n_stocks + 1 + d_implvol + i_bonds)
# invariants to be modeled nonparametrically
ind_nonparametric = list(set(range(i_)) - set(ind_parametric))
db_estimation_nonparametric = {}  # contains the HFP marginals

for i in ind_nonparametric:
    # nonparametric estimation: stocks and S&P 500
    if (db_invariants_nextstep.iloc[0, i] == 'GARCH(1,1)'):
        p_tmp = twist_prob_mom_match(epsi[:, i], 0, 1, p)
        db_estimation_nonparametric[i] = p_tmp

    # nonparametric estimation: implied volatility
    elif (db_invariants_nextstep.iloc[0, i] == 'Random walk'):
        db_estimation_nonparametric[i] = p
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step03): Estimate the marginal distributions for bonds

# +
db_estimation_parametric = {}  # contains the parameters of the t marginals

# invariant series for bonds
epsi_bonds = db_invariants_series.dropna().values
# observations t_bonds for bonds
t_bonds = epsi_bonds.shape[0]
# rescale probabilities to sum to one over bond time frame
p_bonds = p[-t_bonds:] / np.sum(p[-t_bonds:])
# grid of values of degrees of freedom to test
nu_vec = np.arange(nu_min, nu_max + 1)
j_ = len(nu_vec)

for i in ind_parametric:
    # parametric estimation (Student t): bonds
    if (db_invariants_nextstep.iloc[0, i] == 'AR(1)'):
        # time series has missing values
        mu_nu = np.zeros(j_)
        sig2_nu = np.zeros(j_)
        llike_nu = np.zeros(j_)  # log-likelihood

        # fit student t to marginals for a grid of values for nu
        for j in range(j_):
            nu = nu_vec[j]
            # fit Student t model
            mu_nu[j], sig2_nu[j] = fit_locdisp_mlfp(epsi_bonds[:, i],
                                                    p=p_bonds, nu=nu)
            # compute log-likelihood of Student t distribution
            llike_nu[j] = np.sum(p_bonds * (np.log(np.sqrt(sig2_nu[j])) +
                                            tstu.logpdf(epsi_bonds[:, i], nu, mu_nu[j],
                                                        np.sqrt(sig2_nu[j]))))

        # choose nu that gives the highest log-likelihood
        j_max = np.argsort(llike_nu)[-1]
        db_estimation_parametric[i] = {'invariant': i,
                                       'nu': nu_vec[j_max],
                                       'mu': mu_nu[j_max],
                                       'sig2': sig2_nu[j_max]}
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step04): Extract grades

# +
epsi_t = np.zeros((t_bonds, i_))
u = np.zeros((t_, i_))

# compute the (realizations of the) grades U
for i in range(i_):
    # nonparametric estimation: stocks, S&P 500 and implied volatility
    if i in ind_nonparametric:
        u[:, i], _, _ = cop_marg_sep(epsi[:, i],
                                     db_estimation_nonparametric[i])

    # parametric estimation (Student t): bonds
    elif i in ind_parametric:
        epsi_t[:, i] = (epsi_bonds[:, i] - db_estimation_parametric[i]['mu']) \
                       / np.sqrt(db_estimation_parametric[i]['sig2'])
        u[-t_bonds:, i] = tstu.cdf(epsi_t[:, i],
                                   db_estimation_parametric[i]['nu'])
        # values must be < 1 for Student t
        u[-t_bonds:, i] = np.minimum(u[-t_bonds:, i], 0.99999999)
        u[:-t_bonds, i] = np.nan
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step05): Estimate the copula

# +
# flexible probabilities for the Student t copula estimation via MLFP
p_copula = p
p_copula_bonds = p[-t_bonds:]

# grid for the degrees of freedom parameter
nu_vec_cop = np.arange(nu_min_copula, nu_max_copula + 1)
l_ = len(nu_vec_cop)

# initialize variables
rho2_copula_vec = np.zeros((i_, i_, l_))
llike_nu = np.zeros(l_)
epsi_tilde = np.zeros((t_, i_, l_))

db_estimation_copula = {}

for l in range(l_):
    # calculate standardized invariants
    for i in range(i_):
        epsi_tilde[:, i, l] = tstu.ppf(u[:, i], nu_vec_cop[l])

    # estimate copula parameters with maximum likelihood
    _, sig2 = \
        fit_locdisp_mlfp_difflength(epsi_tilde[:, :, l],
                                    p=p_copula,
                                    nu=nu_vec_cop[l],
                                    threshold=10 ** -3,
                                    maxiter=1000)

    # shrinkage: factor analysis
    beta, delta2 = factor_analysis_paf(sig2, k_)
    sig2_fa = beta @ beta.T + np.diag(delta2)

    # compute correlation matrix
    rho2_copula_vec[:, :, l], _ = cov_2_corr(sig2_fa)

    # compute log-likelihood at times with no missing values
    llike_nu[l] = np.sum(p_copula_bonds *
                         np.log(mvt_pdf(epsi_bonds, np.zeros(i_),
                                        rho2_copula_vec[:, :, l],
                                        nu_vec_cop[l])))

# choose nu that gives the highest log-likelihood
l_nu = np.argsort(llike_nu)[-1]
db_estimation_copula = {'nu': np.int(nu_vec_cop[l_nu]),
                        'rho2': rho2_copula_vec[:, :, l_nu]}
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step06): Estimate the distribution of the credit structural invariant

# +
ind_credit = np.array(
    [np.where(db_invariants_series.columns == 'stock GE_log_value')[0][0],
     np.where(db_invariants_series.columns == 'stock JPM_log_value')[0][0]]
)

# extract degrees of freedom
nu_credit = np.int(nu_vec_cop[l_nu])

# extract correlation
rho2_credit = rho2_copula_vec[:, ind_credit, l_nu][ind_credit, :]
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step03-implementation-step07): Save databases

# +
# parametric estimation
out = pd.DataFrame(db_estimation_parametric,
                   columns=[db_estimation_parametric[i]['invariant']
                            for i in ind_parametric])
out.to_csv(path + 'db_estimation_parametric.csv')
del out

# nonparametric estimation
out = pd.DataFrame(db_estimation_nonparametric, columns=ind_nonparametric)
out.to_csv(path + 'db_estimation_nonparametric.csv',
           index=False)
del out

# copula degrees of freedom and correlation matrix
out = pd.DataFrame({'nu': pd.Series(db_estimation_copula['nu']),
                    'rho2':
                        pd.Series(db_estimation_copula['rho2'].reshape(-1))})
out.to_csv(path + 'db_estimation_copula.csv',
           index=None)
del out

# credit copula degrees of freedom and correlation matrix
out = pd.DataFrame({'nu_credit': pd.Series(nu_credit),
                    'rho2_credit':
                        pd.Series(rho2_credit.reshape(-1))})
out.to_csv(path + 'db_estimation_credit_copula.csv',
           index=None)
del out

# flexible probabilities
out = pd.DataFrame({'p': p}, index=dates)
out.index.name = 'dates'
out.to_csv(path + 'db_estimation_flexprob.csv')
del out

# market indicator for flexible probabilities
out = pd.DataFrame({'z': z}, index=dates)
out.index.name = 'dates'
out.to_csv(path + 'db_estimation_z.csv')
del out
# -

# ## Plots

# +
plt.style.use('arpm')

# VIX
date_tick = np.arange(0, t_ - 1, 200)
fig1 = plt.figure(figsize=(1280.0 / 72.0, 720.0 / 72.0), dpi=72.0)
ax1 = fig1.add_subplot(311)
plt.plot(dates, z, color=[0, 0, 0], lw=1.15)
plt.title('Market state', fontweight='bold', fontsize=20)
plt.xticks(dates[date_tick], fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([min(dates), max(dates)])
plt.plot(dates, z_star * np.ones(len(dates)), color='red', lw=1.25)
plt.legend(['Market state', 'Target value'], fontsize=17)

# flexible probabilities
myFmt = mdates.DateFormatter('%d-%b-%Y')
ax2 = fig1.add_subplot(312)
plt.bar(dates, p.flatten(), color='gray')
plt.xlim([min(dates), max(dates)])
plt.title('Time and state conditioning flexible probabilities',
          fontweight='bold', fontsize=20)
plt.xticks(dates[date_tick], fontsize=14)
plt.yticks([], fontsize=14)
plt.xlim([min(dates), max(dates)])
ax2.xaxis.set_major_formatter(myFmt)

# flexible probabilities scatter for invariant i_plot
ax3 = fig1.add_subplot(313)
grey_range = np.r_[np.arange(0, 0.6 + 0.01, 0.01), .85]
[color_map, p_colors] = colormap_fp(p, np.min(p), np.max(p),
                                    grey_range, 0, 10, [10, 0])
p_colors = p_colors.T

plt.xticks(dates[date_tick], fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([min(dates), max(dates)])
plt.scatter(dates, epsi[:, i_plot - 1], s=30, c=p_colors, marker='.',
            cmap=color_map)
plt.title(db_invariants_series.columns[i_plot - 1] + ' observation weighting',
          fontweight='bold', fontsize=20)
ax3.xaxis.set_major_formatter(myFmt)
add_logo(fig1, location=1, set_fig_size=False)
fig1.tight_layout()

# marginal distributions

n_bins = 10 * np.log(t_)

hfp = plt.figure(figsize=(1280.0 / 72.0, 720.0 / 72.0), dpi=72.0)
ax = hfp.add_subplot(111)

if i_plot - 1 in ind_parametric:
    # HFP histogram
    f_eps, x_eps = histogram_sp(epsi_bonds[:, i_plot - 1],
                                p=p_bonds, k_=n_bins)
    bar_width = x_eps[1] - x_eps[0]
    plt.bar(x_eps, f_eps.flatten(), width=bar_width, fc=[0.7, 0.7, 0.7],
            edgecolor=[0.5, 0.5, 0.5])

    # Student t fit
    plt.plot(x_eps, np.squeeze(
        tstu.pdf(x_eps, db_estimation_parametric[i_plot - 1]['nu'],
                 db_estimation_parametric[i_plot - 1]['mu'],
                 np.sqrt(db_estimation_parametric[i_plot - 1]['sig2']))))

else:
    # HFP histogram
    f_eps, x_eps = histogram_sp(epsi[:, i_plot - 1],
                                p=db_estimation_nonparametric[i_plot - 1],
                                k_=n_bins)
    bar_width = x_eps[1] - x_eps[0]
    plt.bar(x_eps, f_eps.flatten(), width=bar_width, fc=[0.7, 0.7, 0.7],
            edgecolor=[0.5, 0.5, 0.5])

plt.title(db_invariants_series.columns[i_plot - 1] + ' invariant distribution',
          fontweight='bold', fontsize=20)
plt.xlabel('Invariant', fontsize=17)
add_logo(hfp, location=1, set_fig_size=False)
hfp.tight_layout()

# copula correlation matrix
fig3 = plt.figure(figsize=(1280.0 / 72.0, 720.0 / 72.0), dpi=72.0)
plt.imshow(db_estimation_copula['rho2'])
plt.colorbar()
plt.grid(False)
plt.title('Estimated correlation matrix', fontweight='bold', fontsize=20)
add_logo(fig3, size_frac_x=0.8, location=9, alpha=0.8, set_fig_size=False)
fig3.tight_layout()

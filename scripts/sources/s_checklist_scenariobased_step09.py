# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_checklist_scenariobased_step09 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step09&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-9).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cvxopt

from arpym.portfolio import spectral_index
from arpym.statistics import meancov_sp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step09-parameters)

# +
# indicates which projection to continue from
# True: use copula-marginal projections
# False: use historical projections
copula_marginal = True

v_stocks_min = 200e6  # minimum budget to invest in stocks

if copula_marginal:
    lambda_inf = 1e-9  # minimum value of the parameter lambda
    lambda_sup = 1e-6  # maximum value of the parameter lambda
    lambda_step = 1e-9  # step in the lambda grid
else:
    lambda_inf = 1e-8  # minimum value of the parameter lambda
    lambda_sup = 1e-5  # maximum value of the parameter lambda
    lambda_step = 1e-8  # step in the lambda grid
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step09-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

# Risk drivers identification
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
n_stocks = int(db_riskdrivers_tools['n_stocks'][0])
n_bonds = int(db_riskdrivers_tools.n_bonds[0])
n_ = n_stocks+n_bonds+3

db_v_tnow = pd.read_csv(path+'db_v_tnow.csv')
v_tnow = db_v_tnow.values.squeeze()

# Aggregation
db_aggregation_tools = pd.read_csv(path+'db_aggregation_tools.csv')
v_h_tnow = db_aggregation_tools['v_h_tnow'][0]

if copula_marginal:
    # Projection
    db_projection_tools = pd.read_csv(path+'db_projection_tools.csv')
    j_ = int(db_projection_tools['j_'][0])

    db_scenprob = pd.read_csv(path+'db_scenario_probs.csv')
    p = db_scenprob['p'].values

    # Pricing
    db_pricing = pd.read_csv(path+'db_pricing.csv')
    pi_tnow_thor = db_pricing.values

    # Aggregation
    db_exante_perf = pd.read_csv(path+'db_exante_perf.csv')
    y_h = db_exante_perf.values.squeeze()

    # Ex-ante evaluation
    db_quantile_and_satis = pd.read_csv(path+'db_quantile_and_satis.csv')
    c_es = db_quantile_and_satis['c_es'][0]
else:
    # Projection
    db_projection_tools = pd.read_csv(path+'db_projection_bootstrap_tools.csv')
    j_ = int(db_projection_tools['j_'][0])

    db_scenprob = pd.read_csv(path+'db_scenario_probs_bootstrap.csv')
    p = db_scenprob['p'].values

    # Pricing
    db_pricing = pd.read_csv(path+'db_pricing_historical.csv')
    pi_tnow_thor = db_pricing.values

    # Aggregation
    db_exante_perf = pd.read_csv(path+'db_exante_perf_historical.csv')
    y_h = db_exante_perf.values.squeeze()

    # Ex-ante evaluation
    db_quantile_and_satis = pd.read_csv(path+'db_quantile_and_satis_historical.csv')
    c_es = db_quantile_and_satis['c_es'][0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step09-implementation-step01): Solving the first step of the mean-variance approach

# +
# define set of parameters for mean-variance frontier
lambda_grid = np.arange(lambda_inf, lambda_sup, lambda_step)
l_ = lambda_grid.shape[0]

# compute expectation and covariance of the P&L
exp_pi, cov_pi = meancov_sp(pi_tnow_thor, p)

# set constraints

# equality constraints
# budget constraint: h'*v_tnow = v_h_tnow
a_budget = v_tnow.reshape(1, -1)
b_budget = np.array(v_h_tnow)
# constraint: do not invest in the S&P
a_sp = np.zeros((1, n_))
a_sp[0, n_stocks] = 1
b_sp = np.array(0)
# combine equality constraints
a = cvxopt.matrix(np.r_[a_budget, a_sp])
b = cvxopt.matrix(np.r_[b_budget, b_sp])

# inequality constraints
# holdings must be nonnegative (no short sale)
u_no_short = -np.eye(n_)
v_no_short = np.zeros(n_)
# investment composition constraint: invest at least v_stocks_min in stocks
u_comp = -np.append(v_tnow[:n_stocks],
                      np.zeros(n_bonds+3)).reshape(1, -1)
v_comp = -np.array(v_stocks_min)
# combine inequality constraints
u = cvxopt.matrix(np.r_[u_no_short, u_comp])
v = cvxopt.matrix(np.r_[v_no_short, v_comp])

h_lambda = np.zeros((l_, n_))
expectation = np.zeros(l_)
variance = np.zeros(l_)

cvxopt.solvers.options['show_progress'] = False
for l in range(l_):
    # objective function
    p_opt = cvxopt.matrix(2*lambda_grid[l]*cov_pi)
    q_opt = cvxopt.matrix(-exp_pi)
    # solve quadratic programming problem
    h_lambda[l, :] = np.array(cvxopt.solvers.qp(p_opt, q_opt, u, v,
                                                a, b)['x']).squeeze()

    expectation[l] = exp_pi@h_lambda[l, :].T
    variance[l] = h_lambda[l, :]@cov_pi@h_lambda[l, :].T

# portfolio weights
w_lambda = (h_lambda*v_tnow) / v_h_tnow
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step09-implementation-step02): Solving the second step of the mean-variance approach

# +
# expected shortfall measure of satisfaction
es_pih_lambda = np.zeros(l_)
# indicator function
def indicator(x):
    return (0 <= x and x <= 1-c_es)
# spectrum function
def spectr_es(x):
    return (1 / (1 - c_es)) * indicator(x)
for l in range(l_):
    es_pih_lambda[l], _ = spectral_index(spectr_es, pi_tnow_thor,
                                         p, h_lambda[l, :])

# quasi-optimal portfolio
ind_lambda_star = np.argmax(es_pih_lambda)
lambda_star = lambda_grid[ind_lambda_star]
h_qsi = np.floor(np.round(h_lambda[ind_lambda_star, :], 20))
# satisfaction from quasi-optimal portfolio
es_pih_qsi = es_pih_lambda[ind_lambda_star]
# ex-ante performance of quasi-optimal portfolio
y_h_es_qsi = pi_tnow_thor@h_qsi
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step09-implementation-step03): Save database

# quasi-optimal portfolio
out = {db_v_tnow.columns[i]: h_qsi[i]
       for i in range(len(h_qsi))}
out = pd.DataFrame(out, index = [0])
if copula_marginal:
    out.to_csv(path+'db_final_portfolio.csv', index=False)
else:
    out.to_csv(path+'db_final_portfolio_historical.csv', index=False)
del out

# ## Plots

# +
plt.style.use('arpm')
fig1 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)
f, xp = histogram_sp(y_h, p=p, k_=30)
xp = xp*1e-6
plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.3, .3, .3], edgecolor='k',
       label = 'Current holdings')
f, xp = histogram_sp(y_h_es_qsi, p=p, k_=30)
xp = xp*1e-6
plt.bar(xp, f, width=xp[1]-xp[0], facecolor=[.6, .6, .6, .9],
        edgecolor='k', label = 'Optimal holdings')
plt.title('Optimized portfolio ex-ante P&L distribution',
         fontsize=20, fontweight='bold')
plt.xlabel(r'$Y_h$ (million USD)', fontsize=17)
plt.legend(fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
add_logo(fig1, set_fig_size=False)

fig2, [ax1, ax2] = plt.subplots(2, 1,
                                figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)
xlim = [np.min(variance), np.max(variance)]

plt.sca(ax1)
plt.plot(variance, expectation, lw=1, label='Efficient frontier')
plt.plot(variance[ind_lambda_star], expectation[ind_lambda_star],
         'ro', label ='Optimal holdings')
plt.title('Mean-variance efficient frontier',
         fontsize=20, fontweight='bold')
plt.xlabel('Variance', fontsize=17)
plt.ylabel('Expectation', fontsize=17)
plt.xlim(xlim)
plt.legend(fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.sca(ax2)
instruments = list(db_v_tnow)
colors = cm.get_cmap('Spectral')(np.arange(n_)/n_)[:, :3]
for n in range(n_):
    if n == 0:
        plt.fill_between(variance, w_lambda[:, n],
                         np.zeros(l_), color=colors[n, :],
                         label = instruments[n])
    else:
        plt.fill_between(variance,
                         np.sum(w_lambda[:, :n+1], axis=1),
                         np.sum(w_lambda[:, :n], axis=1), color=colors[n, :],
                         label = instruments[n])
plt.axvline(x=variance[ind_lambda_star], color='k')
plt.title('Portfolio weights', fontsize=20, fontweight='bold')
plt.xlabel('Variance', fontsize=17)
plt.ylabel('Portfolio weights', fontsize=17)
plt.xlim(xlim)
plt.ylim([0,1])
plt.legend(fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
add_logo(fig2, axis = ax1, set_fig_size=False)
plt.tight_layout()

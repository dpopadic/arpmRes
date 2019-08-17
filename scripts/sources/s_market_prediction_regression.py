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

# # s_market_prediction_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_market_prediction_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from arpym.estimation import conditional_fp, cov_2_corr, exp_decay_fp, fit_lfm_lasso,\
    fit_lfm_mlfp, fit_lfm_ols, fit_lfm_ridge, fit_lfm_roblasso
from arpym.statistics import meancov_sp, multi_r2, scoring, smoothing
from arpym.tools import plot_ellipse
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-parameters)

tau_hl_pri = 13*252  # half life for VIX comp. ret. time conditioning
tau_hl_smooth = 2*21  # half life for VIX comp. ret. smoothing
tau_hl_score = 2*21  # half life for VIX comp. ret. scoring
alpha_leeway = 0.6  # probability included in the range centered in z_vix_star
n_plot = 30  # number of stocks to show in plot
nu = 4  # robustness parameter
pri_param_load = 1.5  # the prior parameters in Bayes are = pri_param_load*t_
lambda_lasso = 10**-5  # lasso penalty
lambda_ridge = 10**-6  # ridge penalty
lambda_beta = 10**-5  # lasso penalty in mixed approach
lambda_phi = 4*10**-5  # glasso penalty in mixed approach

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step00): Load data

# +
path_glob = '../../../databases/global-databases/'
equities_path = path_glob + 'equities/db_stocks_SP500/'

# Stocks
db_stocks_sp = pd.read_csv(equities_path + 'db_stocks_sp.csv',
                           header=1,
                           index_col=0, parse_dates=True)
stocks_names = list(db_stocks_sp.columns)
stocks_sectors = pd.read_csv(equities_path + 'db_stocks_sp.csv', header=None,
                             index_col=0).loc['sector'].tolist()
# Sectors
sector_names = ['dates', 'ConsumerDiscretionary', 'ConsumerStaples', 'Energy',
                'Financials', 'HealthCare', 'InformationTechnology',
                'Industrials', 'Materials', 'TelecommunicationServices',
                'Utilities']
db_sector_idx = pd.read_csv(equities_path+'db_sector_idx.csv', index_col=0,
                            usecols=sector_names,
                            parse_dates=True)
sector_names = sector_names[1:]

# VIX (used for time-state conditioning)
vix_path = path_glob + 'derivatives/db_vix/data.csv'
db_vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                     index_col=0, parse_dates=True)

# intersect dates
dates_rd = pd.DatetimeIndex.intersection(db_stocks_sp.index,
                                         db_sector_idx.index)
dates_rd = pd.DatetimeIndex.intersection(dates_rd, db_vix.index)

# update databases
db_stocks_sp = db_stocks_sp.loc[dates_rd, :]
db_sector_idx = db_sector_idx.loc[dates_rd, :]
db_vix = db_vix.loc[dates_rd, :]

dates = dates_rd[1:]
t_ = len(dates)

# values
v = db_stocks_sp.values
s = db_sector_idx.values
vix = db_vix.values[:, 0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step01): Stocks and factors linear returns

# +
x = np.diff(v, axis=0)/v[:-1, :]
n_ = x.shape[1]

z = np.diff(s, axis=0)/s[:-1, :]
k_ = z.shape[1]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step02): Historical estimation

# +
# time and state conditioning on smoothed and scored VIX returns
# state indicator: VIX compounded return realizations
c_vix = np.diff(np.log(vix))
# smoothing
z_vix = smoothing(c_vix, tau_hl_smooth)
# scoring
z_vix = scoring(z_vix, tau_hl_score)
# target value
z_vix_star = z_vix[-1]
# flexible probabilities
p_base = exp_decay_fp(len(dates), tau_hl_pri)
p = conditional_fp(z_vix, z_vix_star, alpha_leeway, p_base)

# HFP location and dispersion
mu_hat, sig2_hat = meancov_sp(x, p)
_, sig2_z_hat = meancov_sp(z, p)

# OLS loadings
_, beta_ols, sig2_u_ols, _ = fit_lfm_ols(x, z, p, fit_intercept=False)
r2_ols = multi_r2(sig2_u_ols, sig2_hat)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step03): Maximum likelihood - GLM with normal assumption

# +
alpha_mlfp_norm, beta_mlfp_norm, sig2_u_mlfp_norm, _ = \
    fit_lfm_mlfp(x, z, p, 10**9)

# compute r-squared
u_mlfp_norm = x - alpha_mlfp_norm - z@beta_mlfp_norm.T
r2_mlfp_norm = multi_r2(sig2_u_mlfp_norm, sig2_hat)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step04): Maximum likelihood - linear discriminant regression with t assumption

# +
alpha_mlfp_t, beta_mlfp_t, sig2_u_mlfp_t, _ = fit_lfm_mlfp(x, z, p, nu)

# compute r-squared
u_mlfp_t = x - alpha_mlfp_t - z@beta_mlfp_t.T
r2_mlfp_t = multi_r2(sig2_u_mlfp_t, sig2_hat)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step05): Bayesian loadings

# +
# Prior
beta_pri = np.zeros((n_, k_))
sig2_z_pri = sig2_z_hat
t_pri = pri_param_load*t_

sig2_pri = np.diag(cov_2_corr(sig2_hat)[1]**2)
nu_pri = pri_param_load*t_

# Posterior
t_pos = t_pri + t_
nu_pos = nu_pri + t_

beta_pos = (t_pri*beta_pri@sig2_z_pri + t_*beta_ols@sig2_z_hat) @\
    np.linalg.solve(t_pri*sig2_z_pri + t_*sig2_z_hat, np.eye(k_))

sig2_z_pos = 1/t_pos*(t_pri*sig2_z_pri + t_*sig2_z_hat)

sig2_pos_load = 1/nu_pos*(t_*sig2_hat + nu_pri*sig2_pri +
                          t_pri*beta_pri@sig2_z_pri@beta_pri.T +
                          t_*beta_ols@sig2_z_hat@beta_ols.T -
                          t_pos*beta_pos@sig2_z_pos@beta_pos.T)

# compute residuals and r-squared
u_bayes = x-z@beta_pos.T-np.mean(x-z@beta_pos.T, axis=0)
r2_bayes = multi_r2(meancov_sp(u_bayes)[1], sig2_hat)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step06): Regularization: lasso

_, beta_lasso, sig2_u_lasso, _ = fit_lfm_lasso(x, z, p, lambda_lasso,
                                               fit_intercept=False)
# compute r-squared
r2_lasso = multi_r2(sig2_u_lasso, sig2_hat)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step07): Regularization: ridge

_, beta_ridge, sig2_u_ridge, _ = fit_lfm_ridge(x, z, p, lambda_ridge,
                                               fit_intercept=False)
# compute r-squared
r2_ridge = multi_r2(sig2_u_ridge, sig2_hat)

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_market_prediction_regression-implementation-step08): Mixed approach

# +
alpha_rmlfp, beta_rmlfp, sig2_u_rmlfp =\
    fit_lfm_roblasso(x, z, p, nu, lambda_beta=lambda_beta,
                     lambda_phi=lambda_phi)

# compute residuals and r-squared
u_rmlfp = x - alpha_rmlfp - z @ beta_rmlfp.T
r2_rmlfp = multi_r2(sig2_u_rmlfp, sig2_hat)
# -

# ## Plots

# +
plt.style.use('arpm')

# Normal GLM
# compute ellipse grids
ell = plot_ellipse(np.zeros(2), sig2_u_mlfp_norm[:2, :2], r=2,
                   display_ellipse=False)

# limits in colorbars
minncov = np.min(sig2_u_mlfp_norm[:n_plot, :n_plot])
maxxcov = np.max(sig2_u_mlfp_norm[:n_plot, :n_plot])
minnbeta = np.min(beta_mlfp_norm[:n_plot, :])
maxxbeta = np.max(beta_mlfp_norm[:n_plot, :])

xlimu = [np.percentile(u_mlfp_norm[:, 0], 5), np.percentile(u_mlfp_norm[:, 0],
         95)]
ylimu = [np.percentile(u_mlfp_norm[:, 1], 5), np.percentile(u_mlfp_norm[:, 1],
         95)]
xlimu = [min(xlimu[0], ylimu[0]), max(xlimu[1], ylimu[1])]
ylimu = xlimu

# covariances
fig1 = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)

# scatter plot with MLFP-ellipsoid superimposed
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax1.set_xlim(xlimu)
ax1.set_ylim(ylimu)
ax1.axis('square')
ax1.scatter(u_mlfp_norm[:, 0], u_mlfp_norm[:, 1], s=5, c=[[0.5, 0.5, 0.5]])
ax1.plot(ell[:, 0], ell[:, 1], 'r', lw=1.5)
ax1.set_xlabel('$U_1$', fontsize=17)
ax1.set_ylabel('$U_2$', fontsize=17)
ax1.set_title('MLFP residuals - normal', fontweight='bold', fontsize=20)

# heatmaps of the loadings and dispersions
ax3 = plt.subplot2grid((1, 3), (0, 1))
sns.heatmap(sig2_u_mlfp_norm[:n_plot, :n_plot],
            cmap='binary',
            xticklabels=stocks_names[:n_plot],
            yticklabels=stocks_names[:n_plot],
            vmin=minncov,
            vmax=maxxcov,
            square=True)
plt.title('$\sigma_U^{2MLFP}$', fontweight='bold', fontsize=20)

ax4 = plt.subplot2grid((1, 3), (0, 2))
sns.heatmap(beta_mlfp_norm[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.title('$b^{MLFP}$\n$r2 = $ %f' % r2_mlfp_norm, fontweight='bold',
          fontsize=20)

add_logo(fig1, axis=ax4, set_fig_size=False, size_frac_x=1/10)
plt.tight_layout()

# Student t discriminant regression and mixed approach
# compute ellipse grids
ell_mlfp = plot_ellipse(np.zeros(2), sig2_u_mlfp_t[:2, :2],
                        r=2, display_ellipse=False)
ell_rmlfp = plot_ellipse(np.zeros(2), sig2_u_rmlfp[:2, :2],
                         r=2, display_ellipse=False)
# limits in colorbars
minncov = np.min(np.c_[sig2_u_mlfp_t[:n_plot, :n_plot],
                       sig2_u_rmlfp[:n_plot, :n_plot]])
maxxcov = np.max(np.c_[sig2_u_mlfp_t[:n_plot, :n_plot],
                       sig2_u_rmlfp[:n_plot, :n_plot]])
minnbeta = np.min(np.c_[beta_mlfp_t[:n_plot, :],
                        beta_rmlfp[:n_plot, :]])
maxxbeta = np.max(np.c_[beta_mlfp_t[:n_plot, :],
                        beta_rmlfp[:n_plot, :]])

xlimu = [min(np.percentile(u_mlfp_t[:, 0], 2),
             np.percentile(u_rmlfp[:, 0], 2)),
         max(np.percentile(u_mlfp_t[:, 0], 98),
             np.percentile(u_rmlfp[:, 0], 98))]
ylimu = [min(np.percentile(u_mlfp_t[:, 1], 2),
             np.percentile(u_rmlfp[:, 1], 2)),
         max(np.percentile(u_mlfp_t[:, 1], 98),
             np.percentile(u_rmlfp[:, 1], 98))]
xlimu = [min(xlimu[0], ylimu[0]), max(xlimu[1], ylimu[1])]
ylimu = xlimu

# covariances
fig2 = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)

# scatter plot with MLFP-ellipsoid superimposed
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax1.set_xlim(xlimu)
ax1.set_ylim(ylimu)
ax1.axis('square')
ax1.scatter(u_mlfp_t[:, 0], u_mlfp_t[:, 1], s=5, c=[[0.5, 0.5, 0.5]])
ax1.plot(ell_mlfp[:, 0], ell_mlfp[:, 1], 'r', lw=1.5)
ax1.set_xlabel('$U_1$', fontsize=17)
ax1.set_ylabel('$U_2$', fontsize=17)
ax1.set_title('MLFP residuals', fontweight='bold', fontsize=20)

# scatter plot with RMLFP-ellipsoid superimposed
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax2.set_xlim(xlimu)
ax2.set_ylim(ylimu)
ax2.axis('square')
ax2.scatter(u_rmlfp[:, 0], u_rmlfp[:, 1], s=5, c=[[0.5, 0.5, 0.5]])
ax2.plot(ell_rmlfp[:, 0], ell_rmlfp[:, 1], 'r', lw=1.5)
ax2.set_xlabel('$U_1$', fontsize=17)
ax2.set_ylabel('$U_2$', fontsize=17)
ax2.set_title('RMLFP residuals', fontweight='bold', fontsize=20)

# heatmaps of the loadings and dispersions
# MLFP
ax3 = plt.subplot2grid((2, 3), (0, 1))
sns.heatmap(sig2_u_mlfp_t[:n_plot, :n_plot],
            cmap='binary',
            xticklabels=stocks_names[:n_plot],
            yticklabels=stocks_names[:n_plot],
            vmin=minncov,
            vmax=maxxcov,
            square=True)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('$\sigma_U^{2MLFP}$', fontweight='bold', fontsize=20)

ax4 = plt.subplot2grid((2, 3), (0, 2))
sns.heatmap(beta_mlfp_t[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.yticks(fontsize=7)
plt.title('$b^{MLFP}$\n$r2 = $ %f' % r2_mlfp_t, fontweight='bold', fontsize=20)

# RMLFP
ax5 = plt.subplot2grid((2, 3), (1, 1))
sns.heatmap(sig2_u_rmlfp[:n_plot, :n_plot],
            cmap='binary',
            xticklabels=stocks_names[:n_plot],
            yticklabels=stocks_names[:n_plot],
            vmin=minncov,
            vmax=maxxcov,
            square=True)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('$\sigma_U^{2RMLFP}$', fontweight='bold', fontsize=20)

ax6 = plt.subplot2grid((2, 3), (1, 2))
sns.heatmap(beta_rmlfp[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.yticks(fontsize=7)
plt.title('$b^{RMLFP}$\n$r2 = $ %f' % r2_rmlfp, fontweight='bold', fontsize=20)

add_logo(fig2, axis=ax6, set_fig_size=False, size_frac_x=1/12)
plt.tight_layout()

# Bayes
# limits in colorbars
minnbeta = np.min(np.c_[beta_pos[:n_plot, :],
                        beta_ols[:n_plot, :]])
maxxbeta = np.max(np.c_[beta_pos[:n_plot, :],
                        beta_ols[:n_plot, :]])

# heatmaps of the loadings and dispersions
fig3 = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)

ax = plt.subplot2grid((1, 2), (0, 0))
sns.heatmap(beta_ols[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.title('$\hat{b}^{OLSFP}$\n$r^2 = $%f' % r2_ols, fontweight='bold',
          fontsize=20)

ax4 = plt.subplot2grid((1, 2), (0, 1))
sns.heatmap(beta_pos[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.title('$b_{pos}$\n$r^2 = $%f' % r2_bayes, fontweight='bold', fontsize=20)

add_logo(fig3, axis=ax4, set_fig_size=False, size_frac_x=1/12)
plt.tight_layout()

# Lasso and ridge
# limits in colorbars
minnbeta = np.min(np.c_[beta_lasso[:n_plot, :],
                        beta_ridge[:n_plot, :],
                        beta_ols[:n_plot, :]])
maxxbeta = np.max(np.c_[beta_lasso[:n_plot, :],
                        beta_ridge[:n_plot, :],
                        beta_ols[:n_plot, :]])

# heatmaps of the loadings and dispersions
fig4 = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)

ax4 = plt.subplot2grid((1, 3), (0, 0))
sns.heatmap(beta_lasso[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.title('$b^{lasso}$\n$r^2 = $%f' % r2_lasso, fontweight='bold', fontsize=20)

ax = plt.subplot2grid((1, 3), (0, 1))
sns.heatmap(beta_ols[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.title('$\hat{b}^{OLSFP}$\n$r^2 = $%f' % r2_ols, fontweight='bold',
          fontsize=20)

ax6 = plt.subplot2grid((1, 3), (0, 2))
sns.heatmap(beta_ridge[:n_plot, :], cmap='bwr',
            xticklabels=sector_names[:k_],
            yticklabels=stocks_names[:n_plot],
            vmin=minnbeta,
            vmax=maxxbeta,
            center=0)
plt.title('$b^{ridge}$\n$r^2 = $%f' % r2_ridge, fontweight='bold', fontsize=20)

add_logo(fig4, axis=ax6, set_fig_size=False, size_frac_x=1/10)
plt.tight_layout()

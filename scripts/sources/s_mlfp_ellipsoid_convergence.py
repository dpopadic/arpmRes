#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# # s_mlfp_ellipsoid_convergence [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_mlfp_ellipsoid_convergence&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMFPellipsoid).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation import effective_num_scenarios, exp_decay_fp,\
    fit_garch_fp, fit_locdisp_mlfp
from arpym.tools import plot_ellipse, colormap_fp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_mlfp_ellipsoid_convergence-parameters)

tau_hl = 10*252  # prior half life
nu = 4.  # degrees of freedom
gamma = 10**(-5)  # MLFP routine convergence threshold

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_mlfp_ellipsoid_convergence-implementation-step00): Upload data

path = \
    '../../../databases/global-databases/equities/db_stocks_SP500/db_stocks_sp.csv'
stocks = pd.read_csv(path, skiprows=[0], index_col=0, parse_dates=True,
                     usecols=['name', 'CSCO', 'GE'], skip_blank_lines=True)
stocks = stocks.dropna(how='any')  # stocks values

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_mlfp_ellipsoid_convergence-implementation-step01): Compute the log-values of the stocks

x_csco = np.log(np.array(stocks.CSCO))
x_ge = np.log(np.array(stocks.GE))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_mlfp_ellipsoid_convergence-implementation-step02): Compute the invariants using a GARCH(1,1) fit

# +
_, _, epsi_csco = fit_garch_fp(np.diff(x_csco))
_, _, epsi_ge = fit_garch_fp(np.diff(x_ge))

epsi = np.array([epsi_csco, epsi_ge]).T
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_mlfp_ellipsoid_convergence-implementation-step03): Set the exp. decay probabilities for MLFP estimation and compute the effective number of scenarios

p = exp_decay_fp(len(epsi_csco), tau_hl)  # exp. decay flexible probabilities
ens = effective_num_scenarios(p)  # effective number of scenarios

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_mlfp_ellipsoid_convergence-implementation-step04): Perform the MLFP estimation

mu_mlfp, sig2_mlfp = fit_locdisp_mlfp(epsi, p=p, nu=nu, threshold=gamma,
                                      print_iter=True)

# ## Plots

# +
plt.style.use('arpm')

plot_dates = np.array(stocks.index)
cm, c = colormap_fp(p, None, None, np.arange(0, 0.81, 0.01), 0, 1, [0.6, 0.2])

fig, ax = plt.subplots(2, 1)

# scatter plot with MLFP ellipsoid superimposed
plt.sca(ax[0])
plt.scatter(epsi[:, 0], epsi[:, 1], 15, c=c, marker='.', cmap=cm)
plt.axis('equal')
plt.xlim(np.percentile(epsi[:, 0], 100*np.array([0.01, 0.99])))
plt.ylim(np.percentile(epsi[:, 1], 100*np.array([0.01, 0.99])))
plt.xlabel('$\epsilon_1$')
plt.ylabel('$\epsilon_2$')
plot_ellipse(mu_mlfp, sig2_mlfp, color='r')
plt.legend(['MLFP ellipsoid'])
plt.title('MLFP ellipsoid of Student t GARCH(1,1) residuals')

# Flexible probabilities profile
plt.sca(ax[1])
plt.bar(plot_dates[1:], p, color='gray', width=1)
plt.yticks([])
plt.ylabel('$p_t$')
ens_plot = 'Eff. num. of scenarios =  % 3.0f' % ens
plt.title('Exponential decay flexible probabilities.  ' + ens_plot)
add_logo(fig, axis=ax[0])
plt.tight_layout()

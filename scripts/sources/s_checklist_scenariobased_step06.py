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

# # s_checklist_scenariobased_step06 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step06&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-6).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-parameters)

# +
# indicates which projection to continue from
# True: use copula-marginal projections
# False: use historical projections
copula_marginal = True

v_h_tinit = 250e6  # budget at time t_init
v_stocks_budg_tinit = 200e6  # maximum budget invested in stock
h_sp = 0  # holding of S&P 500
h_put_spx = 16000  # holding of put options on S&P 500
h_call_spx = 16000  # holding of call options on S&P 500
h_bonds = 22e6  # notional for bonds
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step00): Load data

# +
# Risk drivers identification
path = '../../../databases/temporary-databases/'
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna())
y = db_riskdrivers_tools['y'][0]
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')
t_init = np.datetime64(db_riskdrivers_tools.t_init[0], 'D')
stock_names = db_riskdrivers_tools.stock_names.dropna()

db_v_tnow = pd.read_csv(path+'db_v_tnow.csv')
v_tnow = db_v_tnow.values.squeeze()

db_v_tinit = pd.read_csv(path+'db_v_tinit.csv')
v_tinit = db_v_tinit.values.squeeze()

if copula_marginal:
    # Projection
    db_projection_tools = pd.read_csv(path+'db_projection_tools.csv')
    j_ = int(db_projection_tools.j_.dropna())

    db_scenprob = pd.read_csv(path+'db_scenario_probs.csv')
    p = db_scenprob.p.values

    # Pricing
    db_pricing = pd.read_csv(path+'db_pricing.csv')
    pi_tnow_thor = db_pricing.values
else:
    # Projection
    db_projection_tools = pd.read_csv(path+'db_projection_bootstrap_tools.csv')
    j_ = int(db_projection_tools.j_.dropna())

    db_scenprob = pd.read_csv(path+'db_scenario_probs_bootstrap.csv')
    p = db_scenprob.p.values

    # Pricing
    db_pricing = pd.read_csv(path+'db_pricing_historical.csv')
    pi_tnow_thor = db_pricing.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step01): Determine stock holdings

h_stocks = np.zeros(n_stocks)
for n in range(n_stocks):
    h_stocks[n] = np.floor(1/n_stocks * v_stocks_budg_tinit/v_tinit[n])

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step02): Define holdings

# +
# bonds
h_bonds = np.repeat(h_bonds, n_bonds)

# holdings
h = np.r_[h_stocks,
          h_sp,
          h_call_spx,
          h_put_spx,
          h_bonds]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step03): Determine initial value of cash holding

# cash at time t_init
cash_tinit = v_h_tinit - h.T@v_tinit

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step04): Determine current value of holding

# +
# cash value at t_now
cash_tnow = cash_tinit

# value of holding at t_now
v_h_tnow = h.T@v_tnow + cash_tnow
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step05): Calculcate ex-ante performance

# ex-ante performance (P&L)
y_h = pi_tnow_thor@h

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step06-implementation-step06): Save databases

# +
out = pd.DataFrame({'Y_h': pd.Series(y_h)})
if copula_marginal:
    # ex-ante performance
    out.to_csv(path+'db_exante_perf.csv',
               index=False)
    del out
else:
    # ex-ante performance
    out.to_csv(path+'db_exante_perf_historical.csv',
               index=False)
    del out

# holdings
out = {db_v_tnow.columns[i]: h[i]
       for i in range(len(h))}
out = pd.DataFrame(out, index=[0])
out = out[list(db_v_tnow.columns)]
out.to_csv(path+'db_holdings.csv',
       index=False)
del out

# additional information
out = pd.DataFrame({'v_h_tnow': v_h_tnow,
                    'cash_tnow': cash_tnow},
                    index=[0])
out.to_csv(path+'db_aggregation_tools.csv',
           index=False)
del out
# -

# ## Plots

# plot ex-ante performance
plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)
f, xp = histogram_sp(y_h, p=p, k_=30)
xp_m = xp*1e-6
plt.bar(xp_m, f, width=xp_m[1]-xp_m[0], facecolor=[.3, .3, .3], edgecolor='k')
plt.title('Ex-ante performance', fontsize=20, fontweight='bold')
plt.xlabel(r'$Y_h$ (million USD)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
add_logo(fig, location=1, set_fig_size=False)

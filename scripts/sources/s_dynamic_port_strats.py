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

# # s_dynamic_port_strats [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_dynamic_port_strats&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBCompDynamicStrat).

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from arpym.portfolio import opt_trade_meanvar
from arpym.tools import plot_dynamic_strats, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-parameters)

# +
v_tnow_strat = 10000  # initial budget
v_tnow_risky = 100  # initial value of the risky instrument
v_tnow_rf = 100  # initial value of the risk-free instrument
t_now = 0  # current time
t_hor = 1  # future horizon (in years)

j_ = 1000  # number of scenarios
k_ = 252  # number of time grids
mu = 0.10  # annually expected return on the underlying
sig = 0.40  # annually expected percentage volatility on the stock index
r_rf = 0.02  # risk-free (money market) interest rate

h_risky = 0.5  # ratio of risky instrument for buy and hold strategy
h_rf = 0.5  # ratio of risk-free instrument for buy and hold strategy
lam = 0.8  # power utility coefficient
mult_cppi = 1.6  # CPPI multiplier
gam = 0.7
mult_dc = 2.0
k_strk = 100  # strike price
v_tnow_floor = 8000  # minimum floor

# parameters for the transaction cost
alpha = 0.5  # slippage power
beta = 0.60  # acceleration parameter
delta_q = 0.1  # volume time horizon
eta_ = 0.142  # normalized slippage coefficient
gam_ = 0.314  # normalized permanent impact coefficient
q_ = 1000  # daily average volume
sig_ = 0.0157  # normalized volatility (intraday)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-implementation-step01): Generate scenarios of the risky and risk-free instruments

dt = (t_hor - t_now) / k_  # time grid (in years)
t = np.arange(0, t_hor + dt, dt)
db_t = np.random.randn(j_, k_)
v_t_risky = v_tnow_risky *\
            np.r_['-1', np.ones((j_, 1)),
                  np.exp(np.cumsum((mu - sig ** 2 / 2) * dt + sig * np.sqrt(dt)
                         * db_t, axis=1))]
v_t_rf = v_tnow_rf * np.exp(r_rf * t)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-implementation-step02): Buy and hold strategy

# +
h_t_risky_bh = h_risky * v_tnow_strat / v_tnow_risky * np.ones((j_, k_ + 1))
h_t_rf_bh = h_rf * v_tnow_strat / v_tnow_rf * np.ones((j_, k_ + 1))
v_t_strat_bh = np.zeros((j_, k_ + 1))
w_t_risky_bh = np.zeros((j_, k_ + 1))

for k in range(k_ + 1):
    v_t_strat_bh[:, k] = h_t_risky_bh[:, k] * v_t_risky[:, k] + \
                         h_t_rf_bh[:, k] * v_t_rf[k]
    w_t_risky_bh[:, k] = h_t_risky_bh[:, k] * v_t_risky[:, k] / \
        v_t_strat_bh[:, k]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-implementation-step03): Maximum power utility strategy

# +
v_t_strat_mpu = np.zeros((j_, k_ + 1))
v_t_strat_mpu[:, 0] = v_tnow_strat
w_t_risky_mpu = np.ones((j_, k_ + 1)) * (mu - r_rf) / sig ** 2 / lam
h_t_risky_mpu = np.zeros((j_, k_ + 1))
h_t_rf_mpu = np.zeros((j_, k_ + 1))
c_mpu = np.zeros((j_, k_ + 1))  # transaction costs

for k in range(k_):
    h_t_risky_mpu[:, k] = w_t_risky_mpu[:, k] * v_t_strat_mpu[:, k] / \
                          v_t_risky[:, k]
    h_t_rf_mpu[:, k] = (v_t_strat_mpu[:, k] - h_t_risky_mpu[:, k] *
                        v_t_risky[:, k]) / v_t_rf[k]
    if k > 0:
        h_start_mpu_k = (h_t_risky_mpu[:, k] - h_t_risky_mpu[:, k - 1]) / q_
        c_mpu[:, k] = -v_t_risky[:, k] * \
            opt_trade_meanvar(h_start_mpu_k, 0, q_, alpha, beta, eta_, gam_,
                              sig_, delta_q)[0]
    v_t_strat_mpu[:, k + 1] = v_t_strat_mpu[:, k] + h_t_risky_mpu[:, k] * \
        (v_t_risky[:, k + 1] - v_t_risky[:, k]) + h_t_rf_mpu[:, k] * \
        (v_t_rf[k + 1] - v_t_rf[k]) - c_mpu[:, k]

h_t_rf_mpu[:, -1] = (v_t_strat_mpu[:, -1] - h_t_risky_mpu[:, -1] *
                     v_t_risky[:, -1]) / v_t_rf[-1]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-implementation-step04): Delta hedging strategy

# +
v_t_strat_dh = np.zeros((j_, k_ + 1))
v_t_strat_dh[:, 0] = v_tnow_strat
w_t_risky_dh = np.zeros((j_, k_ + 1))
h_t_risky_dh = np.zeros((j_, k_ + 1))
h_t_rf_dh = np.zeros((j_, k_ + 1))
c_dh = np.zeros((j_, k_ + 1))  # transaction costs

for k in range(k_):
    m_t_k = np.log(v_t_risky[:, k] / k_strk) / np.sqrt(t_hor - t[k])
    d1_k = (m_t_k + (r_rf + sig ** 2 / 2) * np.sqrt(t_hor - t[k])) / sig
    delta = norm.cdf(d1_k, 0, 1)  # option delta
    w_t_risky_dh[:, k] = v_tnow_strat / v_tnow_risky * \
        v_t_risky[:, k] / v_t_strat_dh[:, k] * delta
    h_t_risky_dh[:, k] = w_t_risky_dh[:, k] * v_t_strat_dh[:, k] / \
        v_t_risky[:, k]
    h_t_rf_dh[:, k] = (v_t_strat_dh[:, k] - h_t_risky_dh[:, k] *
                       v_t_risky[:, k]) / v_t_rf[k]
    if k > 0:
        h_start_dh_k = (h_t_risky_dh[:, k] - h_t_risky_dh[:, k - 1]) / q_
        c_dh[:, k] = -v_t_risky[:, k] *\
            opt_trade_meanvar(h_start_dh_k, 0, q_, alpha, beta, eta_, gam_,
                              sig_, delta_q)[0]
    v_t_strat_dh[:, k + 1] = v_t_strat_dh[:, k] + h_t_risky_dh[:, k] * \
        (v_t_risky[:, k + 1] - v_t_risky[:, k]) + h_t_rf_dh[:, k] * \
        (v_t_rf[k + 1] - v_t_rf[k]) - c_dh[:, k]

delta = np.zeros(j_)
delta[v_t_strat_dh[:, -1] > k_strk] = 1
w_t_risky_dh[:, -1] = v_t_risky[:, -1] / v_t_strat_dh[:, -1] * delta
h_t_risky_dh[:, -1] = w_t_risky_dh[:, -1] * v_t_strat_dh[:, -1] / \
                      v_t_risky[:, -1]
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-implementation-step05): Constant proportion portfolio insurance strategy

# +
v_t_floor = v_tnow_floor * np.exp(r_rf * t)  # floor value

v_t_strat_cppi = np.zeros((j_, k_ + 1))
v_t_strat_cppi[:, 0] = v_tnow_strat
w_t_risky_cppi = np.zeros((j_, k_ + 1))
h_t_risky_cppi = np.zeros((j_, k_ + 1))
h_t_rf_cppi = np.zeros((j_, k_ + 1))
c_cppi = np.zeros((j_, k_ + 1))  # transaction costs

for k in range(k_):
    cush_t_k = np.maximum(0, v_t_strat_cppi[:, k] - v_t_floor[k])

    h_t_risky_cppi[:, k] = mult_cppi * cush_t_k / v_t_risky[:, k]
    w_t_risky_cppi[:, k] = h_t_risky_cppi[:, k] * v_t_risky[:, k] / \
        v_t_strat_cppi[:, k]
    h_t_rf_cppi[:, k] = (v_t_strat_cppi[:, k] -
                         h_t_risky_cppi[:, k] * v_t_risky[:, k]) / v_t_rf[k]
    if k > 0:
        h_start_cppi_k = (h_t_risky_cppi[:, k] - h_t_risky_cppi[:, k - 1]) / q_
        c_cppi[:, k] = -v_t_risky[:, k] * \
            opt_trade_meanvar(h_start_cppi_k, 0, q_, alpha, beta, eta_,
                              gam_, sig_, delta_q)[0]
    v_t_strat_cppi[:, k + 1] = v_t_strat_cppi[:, k] + h_t_risky_cppi[:, k] * \
        (v_t_risky[:, k + 1] - v_t_risky[:, k]) + h_t_rf_cppi[:, k] * \
        (v_t_rf[k + 1] - v_t_rf[k]) - c_cppi[:, k]

cush_t_k_ = np.maximum(0, v_t_strat_cppi[:, -1] - v_t_floor[-1])
w_t_risky_cppi[:, -1] = mult_cppi * cush_t_k_ / v_t_strat_cppi[:, -1]
h_t_risky_cppi[:, -1] = w_t_risky_cppi[:, -1] * v_t_strat_cppi[:, -1] / \
                        v_t_risky[:, -1]
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_dynamic_port_strats-implementation-step06): Constant proportion drawdown control strategy

# +
v_t_strat_dc = np.zeros((j_, k_ + 1))
v_t_strat_dc[:, 0] = v_tnow_strat
w_t_risky_dc = np.zeros((j_, k_ + 1))
h_t_risky_dc = np.zeros((j_, k_ + 1))
h_t_rf_dc = np.zeros((j_, k_ + 1))
c_dc = np.zeros((j_, k_ + 1))  # transaction costs
hwm_t_k = np.zeros(j_)  # high water mark

for k in range(k_):
    hwm_t_k = np.maximum(hwm_t_k, v_t_strat_dc[:, k])
    w_t_risky_dc[:, k] = mult_dc * (v_t_strat_dc[:, k] - gam * hwm_t_k) / \
        v_t_strat_dc[:, k]
    h_t_risky_dc[:, k] = w_t_risky_dc[:, k] * v_t_strat_dc[:, k] / \
        v_t_risky[:, k]
    h_t_rf_dc[:, k] = (v_t_strat_dc[:, k]-h_t_risky_dc[:, k] *
                       v_t_risky[:, k]) / v_t_rf[k]
    if k > 0:
        h_start_dc_k = (h_t_risky_dc[:, k] - h_t_risky_dc[:, k - 1]) / q_
        c_dc[:, k] = -v_t_risky[:, k] * \
            opt_trade_meanvar(h_start_dc_k, 0, q_, alpha, beta, eta_, gam_,
                              sig_, delta_q)[0]
    v_t_strat_dc[:, k + 1] = v_t_strat_dc[:, k] + h_t_risky_dc[:, k] * \
        (v_t_risky[:, k + 1] - v_t_risky[:, k]) + h_t_rf_dc[:, k] * \
        (v_t_rf[k + 1] - v_t_rf[k]) - c_dc[:, k]

hwm_t_k_ = np.maximum(hwm_t_k, v_t_strat_dc[:, -1])
w_t_risky_dc[:, -1] = mult_dc * (v_t_strat_dc[:, -1] - gam * hwm_t_k_) / \
                      v_t_strat_dc[:, -1]
h_t_risky_dc[:, -1] = w_t_risky_dc[:, -1] * v_t_strat_dc[:, -1] / \
                      v_t_risky[:, -1]
# -

# ## Plots

# +
num = 100  # number of selected scenarios
j_sel = -1  # selected scenario

plt.style.use('arpm')

# buy and hold strategy
fig1, _ = plot_dynamic_strats(t, v_t_strat_bh, v_t_risky, w_t_risky_bh,
                              h_t_risky_bh, num, j_sel)
add_logo(fig1, size_frac_x=1/8)

# maximum power utility strategy
fig2, _ = plot_dynamic_strats(t, v_t_strat_mpu, v_t_risky, w_t_risky_mpu,
                              h_t_risky_mpu, num, j_sel)
add_logo(fig2, size_frac_x=1/8)

# delta hedge strategy
fig3, _ = plot_dynamic_strats(t, v_t_strat_dh, v_t_risky, w_t_risky_dh,
                              h_t_risky_dh, num, j_sel)
add_logo(fig3, size_frac_x=1/8)

# CPPI strategy
fig4, _ = plot_dynamic_strats(t, v_t_strat_cppi, v_t_risky, w_t_risky_cppi,
                              h_t_risky_cppi, num, j_sel)
add_logo(fig4, size_frac_x=1/8)

# drawdown control strategy
fig5, _ = plot_dynamic_strats(t, v_t_strat_dc, v_t_risky, w_t_risky_dc,
                              h_t_risky_dc, num, j_sel)
add_logo(fig5, size_frac_x=1/8)

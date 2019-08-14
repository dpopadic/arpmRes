# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter


def tick_label_func(y, pos=None):
    return '%1.f' % (5 * y * 1e-2 // 5)


def tick_label_func_1(y, pos=None):
    return '%0.0f' % y


def plot_dynamic_strats(t, v_t_strat, v_t_risky, w_t_risky, h_t_risky,
                        num, j_sel):
    """For details, see here.

    Parameters
    ----------
        t : array, shape (t_,)
        v_t_strat : array, shape (j_,t_)
        v_t_risky : array, shape (j_,t_)
        w_t_risky : array, shape (j_,t_)
        h_t_risky: array, shape (j_,t_)
        num: int
        j_sel: int



    """

    # adjust v_t_risky so that it has the same initial value as v_t_strat
    v_t_risky = v_t_risky * v_t_strat[0, 0] / v_t_risky[0, 0]

    mu_risky = np.mean(v_t_risky, axis=0, keepdims=True).reshape(-1)
    sig_risky = np.std(v_t_risky, axis=0, keepdims=True).reshape(-1)
    mu_strat = np.mean(v_t_strat, axis=0, keepdims=True).reshape(-1)
    sig_strat = np.std(v_t_strat, axis=0, keepdims=True).reshape(-1)

    plt.style.use('arpm')
    fig = plt.figure()
    gs = GridSpec(1, 2)
    gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0])

    num_bins = int(round(100 * np.log(v_t_strat.shape[1])))
    lgrey = [0.8, 0.8, 0.8]  # light grey
    dgrey = [0.4, 0.4, 0.4]  # dark grey

    j_ = v_t_risky.shape[0]

    x_min = t[0]
    x_max = 1.25 * t[-1]
    y_min = v_t_strat[0, 0] / 4
    y_max = v_t_strat[0, 0] * 2.25

    # scatter plot
    ax4 = plt.subplot(gs[1])
    plt.scatter(v_t_risky[:, -1], v_t_strat[:, -1], marker='.', s=2)
    so = np.sort(v_t_risky[:, -1])
    plt.plot(so, so, label='100% risky instrument', color='r')
    plt.plot([y_min, v_t_risky[j_sel, -1], v_t_risky[j_sel, -1]],
             [v_t_strat[j_sel, -1], v_t_strat[j_sel, -1], y_min], 'b--')
    plt.plot(v_t_risky[j_sel, -1], v_t_strat[j_sel, -1], 'bo')
    ax4.set_xlim(y_min, y_max)
    ax4.set_ylim(y_min, y_max)
    ax4.xaxis.set_major_formatter(FuncFormatter(tick_label_func))
    ax4.yaxis.set_major_formatter(FuncFormatter(tick_label_func))
    plt.xlabel('Strategy')
    plt.ylabel('Risky instrument')
    plt.legend()

    # weights and holdings
    ax3 = plt.subplot(gs1[2])
    y_min_3 = np.min(h_t_risky[j_sel, : -1])
    y_max_3 = np.max(h_t_risky[j_sel, : -1])
    plt.sca(ax3)
    plt.plot(t, w_t_risky[j_sel, :], color='b')
    plt.axis([x_min, x_max, 0, 1])
    plt.xticks(np.linspace(t[0], 1.2 * t[-1], 7))
    plt.yticks(np.linspace(0, 1, 3), color='b')
    plt.ylabel('Weights', color='b')
    plt.xlabel('Time')

    ax3_2 = ax3.twinx()
    plt.plot(t, h_t_risky[j_sel, :], color='black')
    plt.ylabel('Holdings', color='black')
    plt.axis([x_min, x_max, y_min_3 - 1, y_max_3 + 1])
    plt.yticks(np.linspace(y_min_3, y_max_3, 3))
    ax3_2.yaxis.set_major_formatter(FuncFormatter(tick_label_func_1))

    ax1 = plt.subplot(gs1[0], sharex=ax3, sharey=ax4)
    # simulated path, standard deviation of strategy
    for j in range(j_ - num, j_):
        plt.plot(t, v_t_strat[j, :], color=lgrey)
    plt.plot(t, v_t_strat[j_sel, :], color='b')
    plt.plot(t, mu_strat + sig_strat, color='orange')
    plt.plot(t, mu_strat - sig_strat, color='orange')
    plt.xticks(np.linspace(t[0], 1.2 * t[-1], 7))
    # histogram
    y_hist, x_hist = np.histogram(v_t_strat[:, -1], num_bins)
    scale = 0.25 * t[-1] / np.max(y_hist)
    y_hist = y_hist * scale
    plt.barh(x_hist[: -1], y_hist, height=(max(x_hist) - min(x_hist)) /
                                          (len(x_hist) - 1), left=t[-1], facecolor=dgrey, edgecolor=dgrey)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('Strategy')
    ax1.set_ylim(y_min, y_max)
    ax1.yaxis.set_major_formatter(FuncFormatter(tick_label_func))

    # risky instrument
    ax2 = plt.subplot(gs1[1], sharex=ax3, sharey=ax4)
    # simulated path, standard deviation of risky instrument
    for j in range(j_ - num, j_):
        plt.plot(t, v_t_risky[j, :], color=lgrey)
    plt.plot(t, v_t_risky[j_sel, :], color='b')
    plt.plot(t, mu_risky + sig_risky, color='orange')
    plt.plot(t, mu_risky - sig_risky, color='orange')
    plt.xticks(np.linspace(t[0], 1.2 * t[-1], 7))
    # histogram
    y_hist, x_hist = np.histogram(v_t_risky[:, -1], num_bins)
    scale = 0.25 * t[-1] / np.max(y_hist)
    y_hist = y_hist * scale
    plt.barh(x_hist[: -1], y_hist, height=(max(x_hist) - min(x_hist)) /
                                          (len(x_hist) - 1), left=t[-1], facecolor=dgrey, edgecolor=dgrey)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('Risky instrument')
    ax2.set_ylim(y_min, y_max)
    ax2.yaxis.set_major_formatter(FuncFormatter(tick_label_func))

    plt.grid(True)
    plt.tight_layout()

    return fig, gs

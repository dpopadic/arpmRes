#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from arpym.statistics.cop_marg_sep import cop_marg_sep
from arpym.statistics.schweizer_wolff import schweizer_wolff
from arpym.tools.histogram2d_sp import histogram2d_sp
from arpym.tools.logo import add_logo


def invariance_test_copula(epsi, lag_, k_=None,
                           title='Copula invariance test'):
    """For details, see here.

    Parameters
    ----------
        epsi : array, shape (t_,)
        lag_ : scalar
        k_: int
        title : string

    Returns
    -------
        sw: array, shape(lag_,)

    """

    t_ = epsi.shape[0]

    # Step 1: Compute Schweizer-Wolff dependence for lags
    sw = np.zeros(lag_)
    for l in range(lag_):
        sw[l] = schweizer_wolff(np.column_stack((epsi[(l + 1):], epsi[: - (l + 1)])))

    # Step 2: Compute grades scenarios
    x_lag = epsi[:-lag_]
    y_lag = epsi[lag_:]
    # compute grades
    u, _, _ = cop_marg_sep(np.column_stack((x_lag, y_lag)))

    # Step 3: Calculate normalized histogram
    if k_ is None:
        k_ = np.floor(np.sqrt(7 * np.log(t_)))
    f, xi_1, xi_2 = histogram2d_sp(u, k_=k_)

    # Plots
    plt.style.use('arpm')
    fig = plt.figure(figsize=(1280.0 / 72.0, 720.0 / 72.0), dpi=72.0)

    # 2D histogram
    ax = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2,
                          projection='3d')
    plt.sca(ax)
    ax.view_init(30, 45)

    # adjust bin centers to left edges for plotting
    dx = xi_1[1] - xi_1[0]
    dy = xi_2[1] - xi_2[0]
    xpos, ypos = np.meshgrid(xi_1 - dx / 2, xi_2 - dy / 2)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')

    ax.bar3d(xpos, ypos, np.zeros_like(xpos), dx, dy, f.flatten())
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.zaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.invert_xaxis()

    plt.xlabel('Grade obs.', labelpad=10, fontsize=17)
    plt.ylabel('Grade lagged obs.', labelpad=10, fontsize=17)
    plt.title(title, fontsize=20, fontweight='bold', y=1.02)

    # dependence plot
    orange = [.9, .4, 0]
    ax = plt.subplot2grid((3, 3), (1, 2))
    plt.sca(ax)
    xx = range(1, lag_ + 1)
    plt.bar(xx, sw, 0.5, faceColor=[.8, .8, .8], edgecolor='k')
    plt.bar(xx[lag_ - 1], sw[lag_ - 1], 0.5, facecolor=orange,
            edgecolor='k')  # highlighting the last bar
    plt.xlabel('Lag', fontsize=17)
    plt.ylabel('Dependence', fontsize=17)
    plt.ylim([0, 1])
    plt.xticks(np.arange(1, lag_ + 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    add_logo(fig, set_fig_size=False)
    plt.tight_layout()

    return sw

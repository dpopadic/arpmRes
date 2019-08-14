#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_kstest(y1, y2, y_grid, cdf_1, cdf_2, up_band, low_band,
                pos=None, name='Invariance Test', bound=(0, 0)):
    """For details, see here.

    Parameters
    ----------
        y1 : array, shape (~t_/2,)
        y2 : array, shape (~t_/2,)
        y_grid : array, shape (10001,)
        cdf_1 : array, shape (~t_end/2,)
        cdf_2 : array, shape (~t_end/2,)
        up_band : array, shape (10001,)
        low_band : array, shape (10001,)
        pos : dict
        name : string
        bound : tuple, shape (1x2)

    Returns
    -------
    None

    """

    if pos is None:
        pos = {}
        pos[1] = [0.1300, 0.74, 0.3347, 0.1717]
        pos[2] = [0.5703, 0.74, 0.3347, 0.1717]
        pos[3] = [0.1300, 0.11, 0.7750, 0.5]
        pos[4] = [0.3, 1.71]

    # colors
    blue = [0.2, 0.2, 0.7]
    l_blue = [0.2, 0.6, 0.8]
    orange = [.9, 0.6, 0]
    d_orange = [0.9, 0.3, 0]

    # max and min value of the first reference axis settings,
    # for both plots [0] and [1]
    if bound[0] != 0:
        xlim_1 = bound[0]
    else:
        xlim_1 = y_grid[0]
    if bound[1] != 0:
        xlim_2 = bound[1]
    else:
        xlim_2 = y_grid[-1]

    y = np.union1d(y1, y2)

    # max value for the second reference axis setting, for plot [0]
    ycount, _ = np.histogram(y, int(round(10*np.log(len(y.flatten())))),
                             normed=False)
    ylim = np.max(ycount)

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # # plot histogram of Sample 1, y1
    sns.distplot(y1, bins=int(round(10 * np.log(len(y1.flatten())))),
                 kde=False, color=orange,
                 hist_kws={"alpha": 1, "edgecolor": "k"}, ax=ax1)
    ax1.set_xlabel('Sample1')
    ax1.set_xlim((xlim_1, xlim_2))
    ax1.set_ylim([0, ylim*0.8])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax1.grid(False)

    sns.distplot(y2, bins=int(round(10 * np.log(len(y2.flatten())))),
                 kde=False, color=l_blue,
                 hist_kws={"alpha": 1, "edgecolor": "k"}, ax=ax2)
    ax2.grid(False)
    ax2.set_xlabel('Sample2')
    ax2.set_xlim((xlim_1, xlim_2))
    ax2.set_ylim([0, ylim*0.8])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    # plot the cdf[s]
    # plot data on the first reference axis

    ax3.scatter(y1, cdf_1, color=d_orange, s=2)
    ax3.scatter(y2, cdf_2, color=blue, s=2)

    sns.rugplot(y1, height=0.025, color=d_orange, ax=ax3)
    sns.rugplot(y2, height=0.025, color=blue, ax=ax3)

    # plot the (upper and lower) band
    ax3.plot(y_grid, up_band, '-', color='k', lw=0.5)
    ax3.plot(y_grid, low_band, '-', color='k', lw=0.5)
    ax3.set_xlabel('data')
    ax3.set_ylabel('cdf')

    ax3.set_xlim([xlim_1, xlim_2])
    ax3.set_ylim([-0.05, 1.05])
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.suptitle(name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

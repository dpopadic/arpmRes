#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# import seaborn as sns

from arpym.statistics.cdf_sp import cdf_sp
from arpym.tools.histogram_sp import histogram_sp


def invariance_test_ks(epsi, *, conf_lev=0.95,
                       title='Kolmogorov-Smirnov test',
                       plot_test=True):
    """For details, see here.

    Parameters
    ----------
        epsi : array, shape (t_, )
        conf_lev : scalar, optional
        title : string, optional
        plot_test : boolean, optional

    Returns
    -------
        conf_int : array, shape(2)

    """

    # Step 1: Generate two random mutually exclusive partitions of observations

    t_ = epsi.shape[0]
    half_t_ = int(np.round(t_ / 2))

    # random permutation of the given vector
    epsi_perm = np.random.permutation(epsi)

    epsi_a = epsi_perm[: half_t_]
    epsi_b = epsi_perm[half_t_:]
    a_ = epsi_a.shape[0]
    b_ = epsi_b.shape[0]

    # Step 2: Compute the hfp cdfs of the two partitions and the KS statistic

    # compute hfp cdf's
    epsi_sort = np.unique(np.sort(epsi))
    cdf_a = cdf_sp(epsi_sort, epsi_a)
    cdf_b = cdf_sp(epsi_sort, epsi_b)

    # compute statistic
    z_ks = np.max(abs(cdf_a - cdf_b))

    # Step 3: Compute the confidence interval

    alpha = 1 - conf_lev
    z = np.sqrt(-np.log(alpha) * (a_ + b_) / (2 * a_ * b_))

    # Step 4: Generate figure

    if plot_test:
        # build the band for Kolmogorov-Smirnov test
        band_mid = 0.5 * (cdf_a + cdf_b)
        band_up = band_mid + 0.5 * z
        band_low = band_mid - 0.5 * z

        # colors
        blue = [0.2, 0.2, 0.7]
        l_blue = [0.2, 0.6, 0.8]
        orange = [.9, 0.6, 0]
        d_orange = [0.9, 0.3, 0]

        # max and min value of the first reference axis settings,
        xlim_1 = np.percentile(epsi, 1.5)
        xlim_2 = np.percentile(epsi, 98.5)

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        # # plot histogram of Sample 1, y1
        # sns.distplot(epsi_a, bins=int(round(10 * np.log(len(epsi_a.flatten())))),
        #             kde=False, color=orange,
        #             hist_kws={"alpha": 1, "edgecolor": "k"}, ax=ax1)
        nx1, cx1 = histogram_sp(epsi_a, k_=int(round(10 * np.log(len(epsi_a.flatten())))))
        ax1.bar(cx1, nx1, cx1[1] - cx1[0], facecolor=orange, edgecolor='k')
        ax1.set_xlabel('Sample1')
        ax1.set_xlim((xlim_1, xlim_2))
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax1.grid(False)

        # sns.distplot(epsi_b, bins=int(round(10 * np.log(len(epsi_b.flatten())))),
        #             kde=False, color=l_blue,
        #             hist_kws={"alpha": 1, "edgecolor": "k"}, ax=ax2)
        nx2, cx2 = histogram_sp(epsi_b, k_=int(round(10 * np.log(len(epsi_a.flatten())))))
        ax2.bar(cx2, nx2, cx2[1] - cx2[0], facecolor=l_blue, edgecolor='k')
        ax2.grid(False)
        ax2.set_xlabel('Sample2')
        ax2.set_xlim((xlim_1, xlim_2))
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

        ylim = np.max(np.r_[nx1, nx2])
        ax1.set_ylim([0, ylim])
        ax2.set_ylim([0, ylim])
        # plot the cdf[s]
        # plot data on the first reference axis

        ax3.scatter(epsi_a, cdf_sp(epsi_a, epsi_a), color=d_orange, s=2)
        ax3.scatter(epsi_b, cdf_sp(epsi_b, epsi_b), color=blue, s=2)

        # shows partitions epsi_a and epsi_b
        ax3.scatter(epsi_a, 0.002 * np.ones(a_), color=d_orange, s=0.5)
        ax3.scatter(epsi_b, 0.002 * np.ones(b_), color=blue, s=0.5)

        # plot the (upper and lower) band
        ax3.plot(epsi_sort, band_up, '-', color='k', lw=0.5)
        ax3.plot(epsi_sort, band_low, '-', color='k', lw=0.5)
        ax3.set_xlabel('data')
        ax3.set_ylabel('cdf')

        ax3.set_xlim([xlim_1, xlim_2])
        ax3.set_ylim([-0.05, 1.05])
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return z_ks, z

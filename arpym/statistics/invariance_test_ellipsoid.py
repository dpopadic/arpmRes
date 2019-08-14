#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

from arpym.statistics.meancov_sp import meancov_sp
from arpym.tools.plot_ellipse import plot_ellipse
from arpym.tools.histogram_sp import histogram_sp


def invariance_test_ellipsoid(epsi, l_, *, conf_lev=0.95, fit=0, r=2,
                              title='Invariance test',
                              bl=None, bu=None, plot_test=True):
    """For details, see here.

    Parameters
    ----------
        epsi : array, shape(t_)
        l_ : scalar
        conf_lev : scalar, optional
        fit : scalar, optional
        r : scalar, optional
        title : string, optional
        bl : scalar, optional
        bu : scalar, optional
        plot_test : boolean, optional

    Returns
    -------
        rho : array, shape(l_)
        conf_int : array, shape(2)

    """

    if len(epsi.shape) == 2:
        epsi = epsi.reshape(-1)

    if bl is None:
        bl = np.percentile(epsi, 0.25)
    if bu is None:
        bu = np.percentile(epsi, 99.75)

    # Settings
    np.seterr(invalid='ignore')
    sns.set_style('white')
    nb = int(np.round(10 * np.log(epsi.shape)))  # number of bins for histograms

    # Step 1: compute the sample autocorrelations

    rho = np.array([st.pearsonr(epsi[:-k] - meancov_sp(epsi[:-k])[0],
                                epsi[k:] - meancov_sp(epsi[k:])[0])[0]
                    for k in range(1, l_ + 1)])

    # Step 2: compute confidence interval

    alpha = 1 - conf_lev
    z_alpha_half = st.norm.ppf(1 - alpha / 2) / np.sqrt(epsi.shape[0])
    conf_int = np.array([-z_alpha_half, z_alpha_half])

    # Step 3: plot the ellipse, if requested

    if plot_test:
        plt.style.use('arpm')
        # Ellipsoid test: location-dispersion parameters
        x = epsi[:-l_]
        epsi = epsi[l_:]
        z = np.concatenate((x.reshape((-1, 1)), epsi.reshape((-1, 1))), axis=1)

        # Step 3: Compute the sample mean and sample covariance and generate figure

        mu_hat, sigma2_hat = meancov_sp(z)

        f = plt.figure()
        f.set_size_inches(16, 9)
        gs = plt.GridSpec(9, 16, hspace=1.2, wspace=1.2)

        # max and min value of the first reference axis settings,
        # for the scatter and histogram plots

        # scatter plot (with ellipsoid)

        xx = x.copy()
        yy = epsi.copy()
        xx[x < bl] = np.NaN
        xx[x > bu] = np.NaN
        yy[epsi < bl] = np.NaN
        yy[epsi > bu] = np.NaN
        ax_scatter = f.add_subplot(gs[1:6, 4:9])
        ax_scatter.scatter(xx, yy, marker='.', s=10)
        ax_scatter.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        plt.xlabel('obs', fontsize=17)
        plt.ylabel('lagged obs.', fontsize=17)
        plot_ellipse(mu_hat, sigma2_hat, r=r, plot_axes=False,
                     plot_tang_box=False, color='orange',
                     line_width=2, display_ellipse=True)
        plt.suptitle(title, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax_scatter.set_xlim(np.array([bl, bu]))
        ax_scatter.set_ylim(np.array([bl, bu]))

        ax = f.add_subplot(gs[7:, 4:9])

        # histogram plot of observations
        xxx = x[~np.isnan(xx)]
        px = np.ones(xxx.shape[0]) / xxx.shape[0]
        nx, cx = histogram_sp(xxx, p=px, k_=nb)
        hist_kws = {'weights': px.flatten(), 'edgecolor': 'k'}
        fit_kws = {'color': 'orange',
                   'cut': 0
                   }
        if fit == 1:  # normal
            sns.distplot(xxx, hist_kws=hist_kws, kde=False, fit=st.norm, ax=ax)
            plt.legend(['Normal fit', 'Marginal distr'], fontsize=14)
        elif fit == 2 and sum(x < 0) == 0:  # exponential
            sns.distplot(xxx, hist_kws=hist_kws, fit_kws=fit_kws,
                         kde=False, fit=st.expon, ax=ax)
            plt.legend(['Exponential fit', 'Marginal distr'], fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        elif fit == 3 and sum(x < 0) == 0:  # Poisson
            ax.bar(cx, nx, cx[1] - cx[0], facecolor=[0.8, 0.8, 0.8],
                   edgecolor='k')
            k = np.arange(x.max() + 1)
            mlest = x.mean()
            plt.plot(k, st.poisson.pmf(k, mlest), 'o', linestyle='-', lw=1,
                     markersize=3, color='orange')
            plt.legend(['Poisson fit', 'Marginal distr.'], loc=1, fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        else:
            ax.bar(cx, nx, cx[1] - cx[0], facecolor=[0.8, 0.8, 0.8],
                   edgecolor='k')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        ax.get_xaxis().set_visible(False)
        ax.set_xlim(np.array([bl, bu]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        ax = f.add_subplot(gs[1:6, 0:3])
        # histogram plot of lagged observations
        yyy = epsi[~np.isnan(yy)]
        py = np.ones(yyy.shape[0]) / yyy.shape[0]
        hist_kws = {'weights': py.flatten(), 'edgecolor': 'k'}
        fit_kws = {'color': 'orange', 'cut': 0}
        ny, cy = histogram_sp(yyy, p=py, k_=nb)
        if fit == 1:
            sns.distplot(yyy, hist_kws=hist_kws, kde=False, fit=st.norm,
                         vertical=True, ax=ax)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        elif fit == 2 and sum(epsi < 0) == 0:
            sns.distplot(yyy, hist_kws=hist_kws, fit_kws=fit_kws,
                         kde=False, fit=st.expon, vertical=True, ax=ax)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        elif fit == 3 and sum(epsi < 0) == 0:
            ax.barh(cy, ny, cy[1] - cy[0], facecolor=[0.8, 0.8, 0.8],
                    edgecolor='k')
            mlest = epsi.mean()
            k = np.arange(epsi.max() + 1)
            plt.plot(st.poisson.pmf(k, mlest), k, 'o', linestyle='-',
                     lw=1, markersize=3, color='orange')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        else:
            ax.barh(cy, ny, cy[1] - cy[0], facecolor=[0.8, 0.8, 0.8],
                    edgecolor='k')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(np.array([bl, bu]))
        ax.invert_xaxis()

        # autocorrelation plot
        ax = f.add_subplot(gs[1:6, 10:])
        xx = np.arange(1, l_ + 1)
        xxticks = xx
        if len(xx) > 15:
            xxticks = np.linspace(1, l_ + 1, 10, dtype=int)
        plt.bar(xx, rho[:l_], 0.5, facecolor=[.8, .8, .8],
                edgecolor='k')
        plt.bar(xx[l_ - 1], rho[l_ - 1], 0.5, facecolor='orange',
                edgecolor='k')  # highlighting the last bar
        plt.plot([0, xx[-1] + 1], [conf_int[0], conf_int[0]], ':k')
        plt.plot([0, xx[-1] + 1], [-conf_int[0],
                                   -conf_int[0]], ':k')
        plt.xlabel('lag', fontsize=17)
        plt.ylabel('Autocorrelation', fontsize=17)
        plt.axis([0.5, l_ + 0.5, -1, 1])
        plt.xticks(xxticks)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return rho, conf_int

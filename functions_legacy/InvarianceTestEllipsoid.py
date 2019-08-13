from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import plot, bar, legend, ylabel, xlabel, xticks
from numpy import arange, ones, percentile, cov, round, log, sqrt, tile, r_, linspace
from numpy.linalg import norm
from scipy.stats import norm, expon, poisson

np.seterr(invalid='ignore')
sns.set_style('white')

from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from ARPM_utils import matlab_percentile
from HistogramFP import HistogramFP


def InvarianceTestEllipsoid(epsi, acf, lag, fit=0, ell_scale=2, pos=None, name='Invariance test', bound=(0, 0)):
    # This function generates a figure showing the ellipsoid test for
    # invariance.
    # INPUTS
    #  epsi      :[vector](1 x t_end) series of (to be tested as such) invariants
    #  acf       :[row vector] vector of autocorrelation coefficients
    #  lag       :[scalar] lag value to focus on
    #  fit       :[scalar] - if fit==1 the observations are tested to be normally distributed
    #                      - if fit==2 the observations are tested to be exponentially distributed
    #                      - if fit==3 the observations are tested to follow a Poisson dstribution
    #  ell_scale :[scalar] scale coefficient for ellipsoid's radius
    #  pos       :[cell] cell array containing the positions of each graph
    #                    - pos{1} -> position of scatter plot
    #                    - pos{2} -> position of the histogram of observations
    #                    - pos{3} -> position of the histogram of lagged observations
    #                    - pos{4} -> position of the autocorrelation plot
    #                    - pos{5} -> position of the legend
    # name       :[string] title of the figure
    # bound      :[vector](1x2) lower and upper values of scatter plot axis

    ## Code

    if pos is None:
        pos = dict()
        pos['scatter'] = [.2, .45, .3866, .43]  # scatter plot
        pos['epsi'] = [.2905, .12, .205, .2157]  # epsi
        pos['epsi_lagged'] = [.085, .45, .1237, .43]  # epsi_lagged
        pos['autocorrelation'] = [.6, .45, .3366, .43]  # autocorrelation
        pos['legend'] = [.085, .228, .11, .1]  # legend

    orange = [.9, .4, 0]
    colscatter = [0.27, 0.4, 0.9]

    lag_ = acf.shape[0]

    # max and min value of the first reference axis settings, for the scatter and histogram plots
    if bound[0] != 0:
        epsi_l = bound[0]
    else:
        epsi_l = matlab_percentile(epsi[0], 0.25)
    if bound[1] != 0:
        epsi_u = bound[1]
    else:
        epsi_u = matlab_percentile(epsi[0], 99.75)
    epsi_lim = [epsi_l, epsi_u]

    # Ellispoid test

    nb = int(round(10 * log(epsi.shape[1])))  # number of bins
    x = epsi[[0], :-lag]
    y = epsi[[0], lag:]

    px = ones((1, x.shape[1])) / x.shape[1]
    option = namedtuple('options', 'n_bins')
    option.n_bins = nb
    nx, cx = HistogramFP(x, px, option)
    py = ones((1, y.shape[1])) / y.shape[1]
    option.n_bins = nb
    ny, cy = HistogramFP(y, py, option)

    f = plt.gcf()
    gs = plt.GridSpec(3, 6)
    # scatter plot (with ellipsoid)
    X = x[0].copy()
    Y = y[0].copy()
    X[X < epsi_lim[0]] = np.NaN
    X[X > epsi_lim[1]] = np.NaN
    Y[Y < epsi_lim[0]] = np.NaN
    Y[Y > epsi_lim[1]] = np.NaN
    ax_scatter = f.add_subplot(gs[1:, 0:2])
    ax_scatter.scatter(X, Y, marker='.', s=10, c=colscatter)
    ax_scatter.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    xlabel('obs')
    ylabel('lagged obs.')
    m = np.mean(r_[x, y], 1, keepdims=True)
    S = cov(r_[x, y])
    PlotTwoDimEllipsoid(m, S, ell_scale, 0, 0, orange, 2, fig=plt.gcf())
    plt.suptitle(name)
    axlims = plt.axis()
    ax = f.add_subplot(gs[0, :2], sharex=ax_scatter)
    hist_kws = {'weights': px.flatten(), 'edgecolor': 'k'}
    fit_kws = {'color': orange,
               'cut': 0
               }
    # histogram plot of observations
    if fit == 1:  # normal
        sns.distplot(x, hist_kws=hist_kws, kde=False, fit=norm, ax=ax)
        legend(['Normal fit', 'Marginal distr'])
    elif fit == 2 and sum(x[0] < 0) == 0:  # exponential
        sns.distplot(x, hist_kws=hist_kws, fit_kws=fit_kws, kde=False, fit=expon, ax=ax)
        legend(['Exponential fit', 'Marginal distr'])
    elif fit == 3 and sum(x[0] < 0) == 0:  # Poisson
        ax.bar(cx[:-1], nx[0], 0.75)
        k = arange(x[0].max() + 1)
        mlest = x[0].mean()
        plt.plot(k, poisson.pmf(k, mlest), 'o', linestyle='-', lw=1, markersize=3, color=orange)
        legend(['Poisson fit', 'Marginal distr.'], loc=1)
        ax_scatter.set_xticks([0, 10, 20])
    else:
        sns.distplot(x.flatten(), norm_hist=0, hist_kws=hist_kws, kde=False, ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.set_xlim(axlims[0:2])

    ax = f.add_subplot(gs[1:, 2], sharey=ax_scatter)
    hist_kws = {'weights': py.flatten(), 'edgecolor': 'k'}
    fit_kws = {'color': orange,
               'cut': 0}
    # histogram plot of lagged observations
    if fit == 1:
        # hf=histfit(y,nb,normal,color=orange)
        sns.distplot(y, hist_kws=hist_kws, kde=False, fit=norm, vertical=True, ax=ax)
    elif fit == 2 and sum(y[0] < 0) == 0:
        sns.distplot(y, hist_kws=hist_kws, fit_kws=fit_kws, kde=False, fit=expon, vertical=True, ax=ax)
        # epdf_grid=arange(npmin(cy)-0.3,npmax(cy)+0.3+0.01,0.01)
        # epdf=expon.pdf(epdf_grid,mean(y))
        # hf[1]=plot(epdf[epdf>10**-3],epdf_grid[epdf>10**-3],color=orange,lw=2)
    elif fit == 3 and sum(y[0] < 0) == 0:
        ax.barh(cy[:-1], ny[0], 0.75)
        mlest = x[0].mean()
        k = arange(x[0].max() + 1)
        plt.plot(poisson.pmf(k, mlest), k, 'o', linestyle='-', lw=1, markersize=3, color=orange)
    else:
        sns.distplot(y.flatten(), norm_hist=0, hist_kws=hist_kws, kde=False, vertical=True, ax=ax)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(axlims[2:])

    # autocorrelation plot
    ax = plt.subplot2grid((5, 8), (1, 5), rowspan=3, colspan=3)
    xx = arange(1, lag + 1)
    xxticks = xx
    if len(xx) > 15:
        xxticks = linspace(1, lag + 1, 10, dtype=int)
    h6 = bar(xx, acf[:lag], 0.5, facecolor=[.8, .8, .8], edgecolor='k')
    h5 = bar(xx[lag - 1], acf[lag - 1], 0.5, facecolor=orange, edgecolor='k')  # highlighting the last bar
    conf = tile(1.96 / sqrt(epsi.shape[1]), len(xx))
    plot(xx, conf, ':k')
    plot(xx, -conf, ':k')
    xlabel('Lag')
    ylabel('Autocorrelation')
    plt.axis([0.5, lag_ + 0.5, -1, 1])
    xticks(xxticks)
    plt.grid(False)

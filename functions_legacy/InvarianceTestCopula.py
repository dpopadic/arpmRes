from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import bar, xlim, ylim, subplots, ylabel, xlabel, title
from numpy import arange, array, ones, floor, round, log, sqrt, r_
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

plt.style.use('seaborn')

from CopMargSep import CopMargSep
from HistogramFP import HistogramFP


def InvarianceTestCopula(Epsi, dep, lag, name='Invariance test'):
    # This function generates a figure showing the copulas-based tests for
    # invariance.
    #  INPUTS
    #   Epsi    : [matrix]  (1 x t_end) series of (to be tested as such) invariants
    #   dep     : [vector]  (1 x lag_) vector of measures of dependence
    #   lag     : [scalar]   lag value to focus on
    #   name    : [string]   title of the figure

    ## Code
    lag_ = len(dep)
    t_ = Epsi.shape[1]

    # grades scenarios
    probs = ones((1, t_ - lag)) / (t_ - lag)
    X = Epsi[[0],: - lag]
    Y = Epsi[[0],lag : ]
    _, _, U = CopMargSep(r_[X, Y], probs)

    # normalize histogram
    nbin = round(sqrt(7*log(t_)))
    option = namedtuple('option', 'n_bins')
    option.n_bins = array([[nbin, nbin]],dtype=int)
    p = ones((1, len(U[0]))) / len(U[0])
    f, xi = HistogramFP(U,p,option)

    fig = plt.gcf()

    ax = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2, projection='3d')
    plt.sca(ax)
    ax.view_init(30, 45)
    xpos, ypos = np.meshgrid(xi[0][:-1], xi[1][:-1])
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.9 * (xi[0][1] - xi[0][0]) * np.ones_like(xpos)
    dy = dx.copy()
    b = ax.bar3d(xpos, ypos, np.zeros_like(xpos), dx, dy, f.flatten(),zorder=0)
    # adjust x-axis scale (the following method must be adapted case by case)
    labels = ['%1.2f' % i for i in xi[0]]
    step = floor(((len(xi[1]) - 2) / 2))
    index = arange(0, len(xi[0]) - 1, step, dtype=int)  # select entries with labels
    xx = xi[0][:-1]+np.diff(xi[0])/2
    plt.xticks(xx[index])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.invert_xaxis()
    # overlay constant plane 1
    xpos, ypos = np.meshgrid(xi[0], xi[1])
    ax.plot_surface(xpos, ypos, ones(xpos.shape), color='yellow', zorder=1)
    xlabel('Grade obs.', labelpad=10)
    ylabel('Grade lagged obs.', labelpad=10)
    title(name)
    # dependence plot
    orange = [.9, .4, 0]
    ax = plt.subplot2grid((3, 3), (1, 2))
    plt.sca(ax)
    xx = range(1, lag + 1)
    h1 = bar(xx, dep[:lag], 0.5, faceColor=[.8, .8, .8], edgecolor='k')
    h2 = bar(xx[lag - 1], dep[lag - 1], 0.5, facecolor=orange, edgecolor='k')  # highlighting the last bar
    xlabel('Lag')
    ylabel('Dependence')
    ylim([0, 1])
    plt.xticks(arange(1, lag_ + 1))
    plt.grid(False)
    plt.tight_layout()

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import histogram, interp, round, log
from numpy import max as npmax

sns.axes_style("white")


def InvarianceTestKolSmirn(epsi, y1, y2, band_int, cdf_1, cdf_2, up_band, low_band, pos=None, name='Invariance Test',
                           bound=(0, 0)):
    # This function returns a figure for the Kolmogorov-Smirnov (IID) test for invariance
    # INPUTS
    #  epsi      :[vector](1 x t_end) series of (to be tested as such) invariants
    #  y1        :[vector](1 x ~t_end/2) first partition of vector epsi
    #  y2        :[vector](1 x ~t_end/2) second partition of vector epsi
    #  band_int  :[row vector] x-axis values of the (upper and lower) band
    #  cdf_1     :[vector](1 x ~t_end/2) empirical cdf of y1
    #  cdf_2     :[vector](1 x ~t_end/2) empirical cdf of y2
    #  up_band   :[row vector] y-axis values of the upper band
    #  low_band  :[row vector] y-axis values of the lower band
    #  pos       :[cell] cell array containing the positions of each graph
    #                    - pos{1} -> position of the histogram of first sample
    #                    - pos{2} -> position of the histogram of second sample
    #                    - pos{3} -> main plot position
    #                    - pos{4} -> title position
    #  name      :[string] title of the figure
    #  bound     :[vector](1x2) lower and upper values of x-axis

    ## Code

    if pos is None:
        pos = {}
        pos[1] = [0.1300, 0.74, 0.3347, 0.1717]
        pos[2] = [0.5703, 0.74, 0.3347, 0.1717]
        pos[3] = [0.1300, 0.11, 0.7750, 0.5]
        pos[4] = [0.3, 1.71]
    # pos [4]=[band_int[0]+(0.5-0.07)@(band_int[-1]-band_int[0]) 1.8]

    # colors
    blue = [0.2, 0.2, 0.7]
    l_blue = [0.2, 0.6, 0.8]
    orange = [.9, 0.6, 0]
    d_orange = [0.9, 0.3, 0]

    # max and min value of the first reference axis settings, for both plots [0] and [1]
    if bound[0] != 0:
        xlim_1 = bound[0]
    else:
        xlim_1 = band_int[0]
    if bound[1] != 0:
        xlim_2 = bound[1]
    else:
        xlim_2 = band_int[-1]

    # max value for the second reference axis setting, for plot [0]
    ycount, _ = histogram(epsi, int(round(10 * log(len(epsi.flatten())))), normed=False)
    ylim = npmax(ycount)

    # # histograms
    # n1y, n1x = histogram(y1, int(round(10 * log(len(y1.flatten())))))
    # n2y, n2x = histogram(y2, int(round(10 * log(len(y2.flatten())))))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # # plot histogram of Sample 1, y1
    sns.distplot(y1, bins=int(round(10 * log(len(y1.flatten())))), kde=False, color=orange,
                 hist_kws={"alpha": 1, "edgecolor": "k"}, ax=ax1)
    ax1.set_xlabel('Sample1')
    ax1.set_xlim((xlim_1, xlim_2))
    ax1.set_ylim([0, ylim * 0.8])
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax1.grid(False)

    sns.distplot(y2, bins=int(round(10 * log(len(y2.flatten())))), kde=False, color=l_blue,
                 hist_kws={"alpha": 1, "edgecolor": "k"}, ax=ax2)
    ax2.grid(False)
    ax2.set_xlabel('Sample2')
    ax2.set_xlim((xlim_1, xlim_2))
    ax2.set_ylim([0, ylim * 0.8])
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    # plot the cdf[s]
    # plot data on the first reference axis

    up_limit_y1 = interp(y1[0], band_int, up_band)
    low_limit_y1 = interp(y1[0], band_int, low_band)
    up_limit_y2 = interp(y2[0], band_int, up_band)
    low_limit_y2 = interp(y2[0], band_int, low_band)

    ax3.scatter(y1, cdf_1, color=d_orange, s=2)
    ax3.scatter(y2, cdf_2, color=blue, s=2)

    sns.rugplot(y1[0], height=0.025, color=d_orange, ax=ax3)
    sns.rugplot(y2[0], height=0.025, color=blue, ax=ax3)

    #
    # # plot the (upper and lower) band
    ax3.plot(band_int, up_band, '-', color='k', lw=0.5)
    ax3.plot(band_int, low_band, '-', color='k', lw=0.5)
    #
    ax3.set_xlabel('data')
    ax3.set_ylabel('cdf')

    ax3.set_xlim([xlim_1, xlim_2])
    ax3.set_ylim([-0.05, 1.05])
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.suptitle(name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


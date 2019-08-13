import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlim, ylim, ylabel, xlabel, title, plot, bar, barh, xticks, yticks, scatter
from numpy import sort, round, log, histogram, max as npmax

plt.style.use('seaborn')


def PlotDynamicStrats(t,V_t_strat,V_t_risky,W_t_risky):
    ## This function generates two figures. The first plots the evolution of a
    # dynamic strategy of two instruments and the weights on the risky asset.
    # The second one is the scatter/histogram plot of the payoffs of the
    # strategy and the risky asset.
    #
    #  INPUTS
    #   t          :[vector] (n_bar x 1) vector of time
    #   V_t        :[matrix] (n_bar x j_bar) portfolio scenarios
    #   V_t_risky  :[matrix] (n_bar x j_bar) risky instrument scenarios
    #   W_t_risky  :[matrix] (n_bar x j_bar) weight of the risky instrument

    # For details on the exercise, see here .

    ## Code

    # adjust V_t_risky so that it has the same initial value as the strategy
    V_t_risky = V_t_risky*V_t_strat[0,0]/V_t_risky[0,0]

    ## Plot the values and the weights

    j = 1  # select one scenario
    y_max=npmax([V_t_strat[:,j], V_t_risky[:,j]])*1.2 # maximum of the y-axis

    fig1 = figure()
    # plot the scenario
    plt.subplot(2,1,1)

    plot(t,V_t_strat[:,j],lw=2.5,color='b')
    plot(t,V_t_risky[:,j],lw=2,color='r')
    plt.axis([0, t[-1], 0, y_max])
    plt.grid(True)
    ylabel('value')
    title('investment (blue) vs underlying (red) value')

    # bar plot of the weights
    plt.subplot(2,1,2)
    bar(t,W_t_risky[:,j],width=t[1]-t[0],color='r', edgecolor='k')
    plt.axis([0, t[-1], 0, 1])
    plt.grid(True)
    xlabel('time')
    ylabel('$')
    title('percentage of underlying in portfolio')
    plt.tight_layout();

    ## Joint scatter/histogram plot for the payoffs

    fig2 = figure()
    NumBins = int(round(10*log(V_t_strat.shape[1])))

    ax = plt.subplot2grid((4,4),(0,0),rowspan=3)
    # histograms
    [n,D]=histogram(V_t_strat[-1,:],NumBins)
    barh(D[:-1],n,height=D[1]-D[0])
    xticks([])
    plt.grid(True)
    y_lim = plt.ylim()

    ax = plt.subplot2grid((4,4),(3,1),colspan=3)
    [n,D]=histogram(V_t_risky[-1,:],NumBins)
    bar(D[:-1],n,width=D[1]-D[0])
    yticks([])
    plt.grid(True)
    x_lim = plt.xlim()

    # scatter plot
    ax = plt.subplot2grid((4,4),(0,1),rowspan=3, colspan=3)
    scatter(V_t_risky[-1,:],V_t_strat[-1,:],marker='.',s=2)

    so=sort(V_t_risky[-1,:])
    plot(so,so,'r')
    xlim(x_lim)
    ylim(y_lim)
    plt.grid(True)
    xlabel('underlying at horizon')
    ylabel('investment at horizon')
    plt.tight_layout();

    return fig1, fig2

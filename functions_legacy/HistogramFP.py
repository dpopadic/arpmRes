from numpy import nan, histogram, sqrt, arange, maximum, zeros, histogram2d, diff, sum as npsum
from scipy.stats import norm

def HistogramFP(Epsi, p, option):
    # This def computes the histogram of the data series contained in
    # Epsi, according to the Flexible Probabilities p associated with each
    # observation.
    #  INPUTS
    #  Epsi    :[vector](1 x t_end) for the 2d-hist or (2 x t_end) for the 3d-hist data series
    #  p       :[vector](1 x t_end) flexible probabilities associated to the data series Epsi
    #  option  :[struct] input specifying the bins' properties fields:
    #            - .n_bins : number of bins or alternatively the centers of bins
    #            - .tau : horizon
    #              .k_ : number of bins
    #  OUTPUTS
    #  f   :[row vector] vector containing the heights of each bin
    #  xi  :[row vector] vector containing the center of each bin

    ## Code
    n_, _ = Epsi.shape

    if 'n_bins' in option._fields:
        n_bins = option.n_bins
        if n_ == 1:
            if isinstance(n_bins, float) or isinstance(n_bins,int):
                _, xi = histogram(Epsi, int(n_bins))
            elif max(n_bins.shape) == 1:      # compute bins' centers
                _, xi = histogram(Epsi, int(n_bins))
            elif max(n_bins.shape) > 1:
                xi = n_bins
            h = diff(xi,1)[0]
            f = zeros((1, max(xi.shape)-1))
            for k in range(max(xi.shape)-1):  # compute histogram
                if k == max(xi.shape)-2:
                    Index = ((Epsi >= xi[k]) & (Epsi <= xi[k+1])).flatten()
                else:
                    Index = ((Epsi >= xi[k]) & (Epsi < xi[k+1])).flatten()
                f[0, k] = npsum(p[0, Index])
            f = f/h

        elif n_ == 2:
            if n_bins.shape ==(2,1) or n_bins.shape == (1,2):   # compute bins' centers
                _, *xi = histogram2d(Epsi[0], Epsi[1],n_bins.flatten())
                h1 = xi[0][1] - xi[0][0]
                h2 = xi[1][1] - xi[1][0]
                f = zeros((max(xi[0].shape) - 1, max(xi[1].shape) - 1))  # compute histogram
                for k1 in range(max(xi[0].shape) - 1):
                    for k2 in range(max(xi[1].shape) - 1):
                        Index = (Epsi[0] >= xi[0][k1] - h1 / 2) & (Epsi[0] < xi[0][k1] + h1 / 2) & (
                        Epsi[1] >= xi[1][k2] - h2 / 2) & (Epsi[1] < xi[1][k2] + h2 / 2)
                        f[k1, k2] = sum(p[0, Index])
            else:
                xi = zeros((2,n_bins.shape[1]))
                xi[0] = n_bins[0]
                xi[1] = n_bins[1]
                h1 = xi[0][1] - xi[0][0]
                h2 = xi[1][1] - xi[1][0]
                f = zeros((max(xi[0].shape), max(xi[1].shape)))  # compute histogram
                for k1 in range(max(xi[0].shape)):
                    for k2 in range(max(xi[1].shape)):
                        Index = (Epsi[0] >= xi[0][k1] - h1 / 2) & (Epsi[0] < xi[0][k1] + h1 / 2) & (
                        Epsi[1] >= xi[1][k2] - h2 / 2) & (Epsi[1] < xi[1][k2] + h2 / 2)
                        f[k1, k2] = sum(p[0, Index])
            f = f/(h1*h2)

    elif 'tau' in option._fields:
        tau = option.tau
        k_ = option.k_
        a = -norm.ppf(10**(-15), 0, sqrt(tau))# compute bins' centers
        h = 2*a/k_
        xi = arange(-a+h,a+h,h)
        f = zeros((1, max(xi.shape)))
        for k in range(max(xi.shape)):# compute histogram
            Index = (Epsi[0] >= xi[k]-h/2) & (Epsi[0] < xi[k]+h/2)
            f[0,k] = sum(p[0,Index])

        f[0,k_-1] = maximum(1-sum(f[0,:-1]),0)
        f = f/h

    return f, xi

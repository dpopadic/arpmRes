from numpy import arange, array, pi, sqrt


def MarchenkoPastur(q,l,sigma2=1):
    # This function returns the Marchenko-Pastur density for the spectrum of
    # the (n_ x n_) covariance matrix computed on a series (n_ x t_end) whose
    # entries are iid random variables with mean 0 and variance sigma2. Notice
    # that the M-P density depends only on the ratio q = t_end/n_
    #  INPUTS
    # q       :[scalar] ratio q = t_end/n_
    # l       :[scalar] coarseness of the density function to be computed
    # sigma2  :[scalar] variance of the generic n-th invariant
    #  OPS
    # x       :[vector](1 x l) abscissas values at which the M-P density is computed
    # y       :[vector](1 x l) ordinate values of the M-P density, computed in x
    # xlim    :[vector](1 x 2) delimiters of the abscissa coordinates

    ## Code

    eps = 1e-9

    xlim = array([(1-1/sqrt(q))**2, (1+1/sqrt(q))**2])*sigma2
    xlim_tmp = [0,0]
    if q > 1:
        xlim_tmp[1] = xlim[1]-eps
        xlim_tmp[0] = xlim[0]+eps
        dx = (xlim_tmp[1]-xlim_tmp[0])/(l-1)
        x = xlim_tmp[0]+dx*arange(l)
        #y = sqrt( 4*x@q/sigma2-(x@q/sigma2+1-q)**2)/( 2*pi@x)
        y = q*sqrt((xlim[1]-x)*(x-xlim[0]))/(2*pi*x*sigma2)
    elif q < 1:
        xlim_tmp[1] = xlim[1]-eps
        xlim_tmp[0] = xlim[0]+eps
        dx = (xlim_tmp[1]-xlim_tmp[0])/(l-2)
        x = xlim_tmp[0]+dx*arange(l-1)
        y = q*sqrt((xlim[1]-x)*(x-xlim[0]))/(2*pi*x*sigma2)
        xlim[0] = 0
        x = [0, x]
        y = [(1-q), y]
    else:
        xlim = array([[0, 4]])*sigma2
        dx = xlim[1]/l
        x = dx*arange(1,l)
        y = sqrt(4*x/sigma2-(x/sigma2)**2)/(2*pi*x)

    return x, y, xlim

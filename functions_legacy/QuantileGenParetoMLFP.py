from numpy import arange


def QuantileGenParetoMLFP(epsi_bar,p_bar,csi,sigma,p_MLFP=None):
    # This function computes the quantile below the threshold for the tail
    # probabilities p_MLFP<p_bar, based on the GPD approximation of the
    # conditional excess distribution.
    #  INPUTS
    #  epsi_bar  :[scalar] threshold
    #  p_bar     :[scalar] probability associated with the threshold epsi_bar
    #  csi       :[scalar] parameter of the GPD
    #  sigma     :[scalar] parameter of the GPD
    #  p_MLFP     :[row vector] probability levels corresponding to quantiles to be computed
    #  OPS
    #  q_MLFP  :[row vector] vector of quantiles
    #  p_MLFP  :[row vector] vector of probability levels corresponding to quantiles

    ## Code
    c=csi
    s=sigma
    th=epsi_bar

    if p_MLFP is None:
        p_MLFP=arange(10**-5,p_bar+(p_bar-10**-5)/9,(p_bar-10**-5)/9)

    q_MLFP=th-((s/c)*(((p_MLFP/p_bar)**(-c))-1))
    return q_MLFP, p_MLFP

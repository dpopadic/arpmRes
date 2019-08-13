from numpy import sum as npsum
from numpy import zeros, sort, argsort, cumsum, r_
from scipy.stats import norm
from SmoothQuantile import SmoothQuantile

def SatisSmoothQuantile(c, Pi, h_tilde, p):
    # This function computes the smooth quantile and the corresponding gradient at a given confidence level
    # associated to a scenario-probability distribution of an ex-ante performance random variable Y.
    # INPUTS
    # c [scalar]:       confidence level
    # Pi [vector]:      n_ x j_ scenarios of the instruments' P&L's
    # h_tilde [vector]: n_ x 1 standardized holdings associated to the ex-ante performance Y
    # p [vector]:       1 x j_ probabilities associated with the scenarios
    # OP
    # q [scalar]:       smooth quantile at the confidence level c
    # grad_q [vector]:  n_x1 gradient of the smooth quantile with respect to h_tilde

    # For details on the exercise, see here .

    ## Code
    Y = h_tilde.T@Pi

    q,w,order = SmoothQuantile(c,Y,p)
    grad_q = Pi[:,order]@w.T

    return q, grad_q

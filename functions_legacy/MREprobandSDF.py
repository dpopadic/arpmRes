from numpy import diag, diagflat
from numpy.linalg import pinv

from MinRelEntFP import MinRelEntFP


def MREprobandSDF(V,v_tnow,p,k):
    #This function computes the minimum entropy numeraire probabilities and the
    #minimum entropy probability stochastic discount factor.
    #INPUTS
    #  V          : [matrix]  n_ x j_ payoff matrix
    #  v_tnow     : [vector]  n_ x 1  vector of curret values
    #  p          : [vector]  1 x j_   vector of probabilities
    #  k          : [scalar] row index of the numeraire in the payoff matrix
    #OPS
    #  p_mre      : [vector] 1 x j_  vector of minimum relative entropy
    #                                probabilities
    #
    # SDF_mre     : [vector] 1 x j_ vector minimum relative entropy Stochastic
    #                               Discount Factor

    # For details on the exercise, see here .

    ## Code
    #check if the index selected is actually a numeraire
    if any(V[k,:]<0):
        print('V[k,:] is not a numeraire.')
        return None

    #compute the views
    Aeq = V@diag(V[k,:]**(-1))
    beq = v_tnow/v_tnow[k]

    #minimum entropy numeraire probability

    p_numer,_ = MinRelEntFP(p,None,None,Aeq,beq)

    SDF_mre = (v_tnow[k]*p_numer.dot(pinv(diagflat(V[k,:].T))))@diagflat(p**-1)

    return SDF_mre,p_numer

from numpy import sum as npsum, zeros, ones, r_

from CrispProbabilities import CrispProbabilities
from MinRelEntFP import MinRelEntFP


def ConditionalFP(Conditioner, prior):
    #Flexible probabilities conditioned via entropy pooling
    # INPUT
    # Conditioner  :[struct] with fields
    #  Series      :[vector] (1 x t_end) time series of the conditioner
    #  TargetValue :[vector] (1 x k_) target values for the conditioner
    #  Leeway:     :[scalar] (alpha) probability contained in the range, which is symmetric around the target value.
    # prior        :[vector] (1 x t_end) prior set of probabilities
    # OUTPUT
    # Fprob        :[vector] (k_ x t_end) conditional flexible probabilities for each of the k_ target values
    ###########################################################################

    z = Conditioner.Series
    zz = Conditioner.TargetValue
    alpha=Conditioner.Leeway

    t_ = z.shape[1]
    k_ = zz.shape[1]

    # CRISP PROBABILITIES
    p,_,_ = CrispProbabilities(Conditioner)
    p[p == 0] = 10**-20

    for i in range(k_):
        p[i, :] = p[i, :]/npsum(p[i, :])

    Fprob=zeros((k_,t_))
    #FLEXIBLE PROBABILITIES
    for i in range(k_):
        #moments
        mu=p[i,:]@z.T
        s2=p[i,:]@(z.T**2)-mu**2

        #constraints
        a=(z**2)
        b=(mu**2)+s2
        aeq = r_[z, ones((1,t_))]
        beq = r_[mu, 1].reshape((-1,1))

        #FP
        Fprob[i, :],_ = MinRelEntFP(prior, a, b, aeq, beq)

    return Fprob
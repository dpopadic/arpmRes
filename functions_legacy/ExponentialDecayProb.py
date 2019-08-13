import matplotlib.pyplot as plt
from numpy import arange, log, exp
from numpy import sum as npsum

plt.style.use('seaborn')


def ExponentialDecayProb(t_,tau_hl):
    #Flexible probabilities computed via exponential decay with half-life
    #tau_hl
    #INPUTS
    # t_end [scalar]: length of the desired vector of probabilities
    # tau_hl [scalar]: half-life of the exponential decay
    #OP
    #p [vector]: (1 x t_end) vector of flexible probabilities obtained via exponential decay
    p=exp(-(log(2)/tau_hl)*arange(t_,1+-1,-1)).reshape(1,-1)
    p=p/npsum(p)
    return p


from collections import namedtuple

from numpy import exp, log, sum as npsum


def EffectiveScenarios(p, Type=None):
    # This def computes the Effective Number of Scenarios of Flexible
    # Probabilities via different types of defs
    #  INPUTS
    #   p       : [vector] (1 x t_) vector of Flexible Probabilities
    #   Type    : [struct] type of def: 'ExpEntropy', 'GenExpEntropy'
    #  OUTPUTS
    #   ens     : [scalar] Effective Number of Scenarios
    # NOTE:
    # The exponential of the entropy is set as default, otherwise
    # Specify Type.ExpEntropy.on = true to use the exponential of the entropy
    # or
    # Specify Type.GenExpEntropy.on = true and supply the scalar
    # Type.ExpEntropy.g to use the generalized exponential of the entropy

    # For details on the exercise, see here .
    if Type is None:
        Type = namedtuple('type',['Entropy'])
        Type.Entropy = 'Exp'
    if Type.Entropy != 'Exp':
        Type.Entropy = 'GenExp'

    ## Code

    if Type.Entropy == 'Exp':
        p[p==0] = 10**(-250)    #avoid log(0) in ens computation
        ens = exp(-p@log(p.T))
    else:
        ens = npsum(p ** Type.g) ** (-1 / (Type.g - 1))

    return ens
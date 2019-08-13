import matplotlib.pyplot as plt
from numpy import tril, ones, zeros, r_, array, int

plt.style.use('seaborn')

from MinRelEntFP import MinRelEntFP

def TwistFPMomMatch(x, p, mu_, s2_=None):
    # This function twists Flexible Probabilities p by Entropy Pooling to match
    # arbitrary moments mu_ sigma2_
    #  INPUTS
    #   x      : [matrix] (n_ x j_) scenarios
    #   p      : [vector] (1 x j_) flexible probabilities
    #   mu_    : [vector] (n_ x 1) target means
    #   s2_    : [matrix] (n_ x n_) target covariances
    #  OPS
    #   p_     : [vector] (1 x j_) twisted flexible probabilities

    # For details on the exercise, see here .

    ## Code

    n_ = x.shape[0]
    j_ = p.shape[1]

    # Step 1. Linear views

    # initialize matrix constraints
    a = ones((1,j_))
    b = 1

    # expectations
    view_mu = mu_
    a = r_[a, x]
    b = r_[array([[b]]), view_mu]

    # second moments
    if s2_ is not None:
        view_mu2 = tril(s2_ + mu_@mu_.T)
        view_mu2 = view_mu2.flatten()
        vech_view_mu2 = view_mu2[view_mu2!=0]
        vech_x2 = zeros((int(n_*(n_+1)/2),j_))
        for j in range(j_):
            x2 = tril(array([x[:,j]@x[:,j].T]))
            #x2 = x2.flatten()
            if n_>1:
                vech_x2[:,j] = x2[x2!=0] # mimic vech operator
            else:
                vech_x2[:,j] = x2
        a = r_[a, vech_x2]
        b = r_[b, array([vech_view_mu2])]

    # Step 2. Twisted flexible probabilties
    p_ = MinRelEntFP(p, None, None, a, b)[0] # entropy pooling
    return p_

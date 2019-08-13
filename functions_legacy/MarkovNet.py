import matplotlib.pyplot as plt
from numpy import sum as npsum, diagflat
from numpy import zeros, sort, where, diag, eye, abs, sqrt
from sklearn.covariance import graph_lasso

from numpy.linalg import pinv

plt.style.use('seaborn')

import numpy as np
np.seterr(invalid='ignore')


def MarkovNet(sigma2, k, lambda_vec, tol=10**-14, opt=0):
    # This function performs parsimonious Markov network shrinkage on
    # correlation matrix c2
    #  INPUTS
    # sigma2      :[matrix](n_ x n_) input matrix
    # k           :[scalar] number of null entries to be reached
    # lambda_vec  :[vector](1 x n_) penalty values
    # tol         :[scalar] tolerance to check the number of null entries
    # opt         :[scalar] if !=0 forces the function to return matrices c2_bar and phi2 computed for each penalty value in lambda_vec
    #  OPS
    # sigma2_bar  :[matrix](n_ x n_) shrunk covariance matrix
    # c2_bar      :[matrix](n_ x n_) shrunk correlation matrix
    # phi2_bar        :[matrix](n_ x n_) inverse of the shrunk correlation matrixc2_bar
    # lambda_bar  :[scalar] optimal penalty value
    # conv        :scalar ==1 if the target of k null entries is reached ==0 otherwise
    # l_bar       :[scalar] if opt!=0 l_bar is the index such that c2_bar((:,:,l_bar)) and phi2(:,:,l_bar) are the optimal matrices

    # For details on the exercise, see here .

    ## Code

    lambda_vec = sort(lambda_vec)

    l_ = len(lambda_vec)

    c2_bar = zeros(sigma2.shape+(l_,))
    phi2_bar = zeros(sigma2.shape+(l_,))
    z = zeros(l_)

    # Compute correlation
    sigma_vec = sqrt(diag(sigma2))
    c2 = diagflat(1/sigma_vec)@sigma2@diagflat(1/sigma_vec)

    for l in range(l_):
        lam = lambda_vec[l]
        # Perform Graphical Lasso
        _,invs2_tilde,*_ = graph_lasso(c2,lam)
        # Correlation extraction
        c2_tilde = eye(sigma2.shape[0]).dot(pinv(invs2_tilde))
        c2_bar[:,:,l] = diagflat(1/diag(sqrt(c2_tilde)))@c2_tilde@diagflat(1/diag(sqrt(c2_tilde)))# estimated correlation matrix
        phi2_bar[:,:,l] = diagflat(diag(sqrt(c2_tilde)))@invs2_tilde@diagflat(diag(sqrt(c2_tilde)))# inverse correlation matrix

        tmp = abs(phi2_bar[:,:,l])
        z[l] = npsum(tmp<tol)

    # Selection
    index = where(z>=k)[0]
    if index == []:
        index = l_
        conv = 0# target of k null entries not reached
    else:
        conv = 1# target of k null entries reached
    l_bar = index[0]
    lambda_bar = lambda_vec[l_bar]

    # Output
    if opt == 0:
        c2_bar = c2_bar[:,:,l_bar]# shrunk correlation
        phi2_bar = phi2_bar[:,:,l_bar]# shrunk inverse correlation
        l_bar = None
        sigma2_bar = diagflat(sigma_vec)@c2_bar@diagflat(sigma_vec)# shrunk covariance
    else:
        sigma2_bar = zeros(sigma2.shape+(l_,))
        for l in range(l_):
            sigma2_bar[:,:,l] = diagflat(sigma_vec)@c2_bar[:,:,l]@diagflat(sigma_vec)

    return sigma2_bar, c2_bar, phi2_bar, lambda_bar, conv, l_bar

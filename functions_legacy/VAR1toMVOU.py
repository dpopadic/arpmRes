import warnings

import numpy as np
from numpy import kron, reshape, real, zeros, eye, abs, log, exp
from numpy.linalg import eig, norm


def VAR1toMVOU(alpha, b, sig2_U, tau):
    # This function converts the 1-step parameters of the VAR(1) process into
    # the parameters of the MVOU process
    #  INPUTS
    #   alpha         : [vector](n_ x 1)  parameter alpha of the VAR(1) process
    #   b             : [matrix](n_ x n_) parameter b of the VAR(1)
    #   sig2_U        : [matrix](n_ x n_) residuals covariance of the VAR(1) process
    #   tau           : [scalar]          time between subsequent observations

    #  OPS
    #   mu            : [vector](n_ x 1)  parameter mu of the MVOU
    #   theta         : [matrix](n_ x n_) transition matrix of the MVOU
    #   sig2          : [matrix](n_ x n_) parameter sig2 of the MVOU
    #   r             : [vector](1 x 2)   relative error for computation of theta and sigma2

    ## Code
    # parameters
    n_ = alpha.shape[0]
    Tol_eig = 10**-10
    beta = b-eye(n_)
    r = zeros((1,2))

    Diag_lambda2, e = eig(beta)

    lambda2_beta = Diag_lambda2 # eigenvalues of beta

    # small eigenvalues of beta are set to 0 to prevent numerical noise
    lambda2_beta[abs(lambda2_beta)<Tol_eig] = 0
    Diag_lambda2 = np.diagflat(lambda2_beta)
    # measure error due to diagonalization + filtering small eigenvalues
    if norm(beta,ord='fro') == 0:
        r[0,0] = 0
    else:
        r[0,0] = norm(np.dot(e@Diag_lambda2,np.linalg.pinv(e))-beta,ord='fro')

    # compute theta in the basis of eigenvectors e
    theta_diag = -log(lambda2_beta + 1)/tau
    # theta_diag = real(theta_diag)

    # warning: eigenvalues of theta, existence of explosive linear combinations of MVOU
    if any(real(theta_diag) < 0):
        warnings.warn('theta has eigenvalues with negative real part: the MVOU is explosive')
    # compute theta in the original basis
    theta = np.dot(e@np.diagflat(theta_diag),np.linalg.pinv(e))
    theta = real(theta)

    # compute mu in the basis of eigenvectors e
    f = zeros((n_,1),dtype=np.complex128)
    f[abs(theta_diag)<=Tol_eig] = 1/tau    # small eigenvalues are set to 1/tau
    num = theta_diag[abs(theta_diag)>Tol_eig]
    denom = (1-exp(-theta_diag[abs(theta_diag)>Tol_eig]*tau))
    f[abs(theta_diag)>Tol_eig,0] = num/denom
    #compute mu in the original basis
    mu = np.dot(e@np.diagflat(f),np.linalg.pinv(e))@alpha
    mu = real(mu)

    # compute sig2 in the basis of eigenvectors e
    kronsum = kron(theta,eye(n_)) + kron(eye(n_),theta)
    Diag_lambda2_kron, e_kron = eig(kronsum)
    lambda2_kron = real(Diag_lambda2_kron)
    Diag_lambda2_kron = np.diagflat(Diag_lambda2_kron)
    # measure error due to diagonalization of kronsum
    if norm(kronsum, ord='fro') == 0:
        r[0,1] = 0
    else:
        r[0,1] = norm(np.dot(e_kron@Diag_lambda2_kron,np.linalg.pinv(e_kron))-kronsum,ord='fro')
    lambda_a = zeros((len(Diag_lambda2_kron),1))
    vec_sig2_U = reshape(sig2_U, (n_**2,1),'F')
    lambda_a[abs(lambda2_kron) <= Tol_eig] = 1/tau
    index = abs(lambda2_kron) > Tol_eig
    lambda_a[index,0] = lambda2_kron[index]/(1-exp(-lambda2_kron[index]*tau))
    # compute sig2 in the original basis
    a = np.dot(e_kron@np.diagflat(lambda_a),np.linalg.pinv(e_kron))
    vec_sigma = a@vec_sig2_U
    sig2 = reshape(vec_sigma,(n_,n_),'F')
    sig2 = real(sig2)

    return mu, theta, sig2, r

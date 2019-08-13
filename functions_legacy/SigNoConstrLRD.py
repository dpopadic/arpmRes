from collections import namedtuple

from numpy import reshape, zeros, diag, eye, sqrt, kron, r_, diagflat, array


def SigNoConstrLRD(theta, a, q, n_, k_, matrix=None):
    # This function computes the constraint function of signal-to-noise ratios
    # with low-rank-diagonal structure on covariance
    #  INPUTS
    # theta   : [vector] (n_*(2 + k_) x 1) vector of variables: theta = (mu, beta.flatten(), d)
    # a       : [matrix] (m_ x n_) matrix defining constraint function
    # q       : [vector] (m_ x 1)  vector defining constraint function
    # n_      : [scalar] market dimension
    # k_      : [scalar] number of factors
    # matrix  : [struct] constant matrices for derivatives (optional)
    #  OPS
    # constr  : [vector] (m_ x 1) constraint values
    # grad    : [matrix] (n_*(2 + k_) x m_) gradient
    # hess    : [matrix] (n_*(2 + k_) x n_*(2 + k_)*m_) hessian

    # For details on the exercise, see here .

    if matrix is None:
        matrix = namedtuple('matrix', ['hm' 'hm2', 'km1'])
        matrix.hm = array([])
        matrix.hm2 = array([])
        matrix.km1 = array([])

    ## Code
    mu, s2, beta, d = theta2param(theta, n_, k_)
    # constraint
    constr = a@(mu.flatten() / sqrt(diag(s2))) - q

    i_n = eye((n_))
    alpha = mu/diag(s2).reshape(-1,1)**(3/2)

    if matrix.hm.size==0:
        matrix.hm = diag(i_n.flatten())

    const1 = sqrt(diag(diag(1 / s2)))
    const2 = kron(beta.T, i_n)@matrix.hm
    const3 = diagflat(alpha * d)

    # gradient
    grad_mu = const1@a.T
    grad_b = -const2@kron(a, alpha.T).T
    grad_d = -const3@a.T
    grad = r_[grad_mu.reshape(-1,1), grad_b, grad_d.reshape(-1,1)]

    m_, _ = a.shape
    i_k = eye((k_))

    if matrix.hm2.size==0:
        matrix.hm2 = zeros((n_,n_**2))
        for n in range(n_):
            matrix.hm2 = matrix.hm2 + kron(i_n[:, [n]].T, diagflat(i_n[:, [n]]))

    if matrix.km1.size==0:
        matrix.km1 = zeros((k_*n_,k_*n_**2))
        for n in range(n_):
            matrix.km1 = matrix.km1 + kron(kron(i_n[:, [n]].T, i_k), diagflat(i_n[:, [n]]))

    const1 = diag(diag(1 / s2)) ** (3/2)
    const2 = kron(beta.T, i_n)
    const3 = const2@matrix.hm
    const4 = kron(a.T, const1)

    const6 = diag(s2).reshape(-1,1) ** (3/2)
    const7 = diag(s2).reshape(-1,1) ** (5/2)
    const8 = -kron(d.T, i_n)@matrix.hm
    const9 =3*const3@kron(beta, diagflat(mu / const7)) - kron(i_k, diagflat(mu / const6))
    const10 = diagflat((mu * d) / const7)
    const11 = diagflat(mu * ((3*d ** 2)/ const7 - 1 / const6))

    # hessian
    grad2_mumu= zeros((n_, n_*m_))
    grad2_bmu = -const3@const4
    grad2_dmu = const8@const4
    grad2_bb = matrix.km1@kron(a.T, const9)
    grad2_bd =3 * const3@kron(a.T, const10)
    grad2_dd = matrix.hm2@kron(a.T, const11)

    for m in range(1,m_+1):
        hess_mumu = grad2_mumu[:, (m-1)*n_:m*n_+1]
        hess_bmu=grad2_bmu[:, (m-1)*n_:m*n_+1]
        hess_dmu=grad2_dmu[:, (m-1)*n_:m*n_+1]
        hess_bb=grad2_bb[:, (m-1)*n_*k_:m*n_*k_+1]
        hess_bd=grad2_bd[:, (m-1)*n_:m*n_+1]
        hess_dd=grad2_dd[:, (m-1)*n_:m*n_+1]

        h = r_[r_['-1',hess_mumu,  hess_bmu.T,   hess_dmu.T],
                  r_['-1',hess_bmu,   hess_bb,     hess_bd],
                     r_['-1',hess_dmu,   hess_bd.T,    hess_dd]]
        if m ==1:
            hess = h.copy()
        else:
            hess = r_['-1',hess, h]

    # return constr, grad, hess
    return constr, grad, hess


def theta2param(theta, n_, k_):
    id = range(n_)
    mu = reshape(theta[id], (-1, 1),'F')
    id = range(n_, n_ + n_*k_)
    b  = reshape(theta[id], (n_, k_),'F')
    id = range(n_+ n_*k_, n_*(2 + k_))
    d = reshape(theta[id], (-1, 1),'F')

    s2 = b@b.T + diagflat(d**2)
    return mu, s2, b, d

import matplotlib.pyplot as plt
from numpy import zeros, r_
from numpy.linalg import solve

plt.style.use('seaborn')

from ARPM_utils import nullspace
from CopMargSep import CopMargSep
from CopMargComb import CopMargComb
from FPmeancov import FPmeancov


def CopulaOpinionPooling(X_pri, p, v, c, FZ_pos):
    # This function performs the Copula Opinion Pooling approach for distributional
    # views processing
    #  INPUTS
    #   X_pri       : [matrix] (n_ x j_) market prior scenarios
    #   p           : [vector] (1 x j_) Flexible Probabilities
    #   v           : [matrix] (k_ x n_) pick matrix
    #   c           : [vector] (k_ x 1) confidence levels
    #   FZ_pos      : [cell] (k_ x 1) views cdf's
    #  OPS
    #   X_pos       : [matrix] (n_ x j_) market updated scenarios
    #   Z_pri       : [matrix] (k_ x j_) view variables prior scenarios
    #   U_pri       : [matrix] (k_ x j_) copula of prior view variables
    #   Z_pos       : [matrix] (k_ x j_) updated scenarios of view variables
    #   v_tilde     : [matrix] (n_ x n_) augmented pick matrix
    #   Z_tilde_pri : [matrix] (n_x j_) augmented prior scenarios view variables
    #   Z_tilde_pos : [matrix] (n_x j_) augmented posterior scenarios view variables

    # For details on the exercise, see here .
    ## Code

    [_,j_]=X_pri.shape
    k_=v.shape[0]

    # scenarios of the prior view variables
    Z_pri = v@X_pri

    # copula of the view variables
    Z_sorted, FZ_, U_pri = CopMargSep(Z_pri,p) # copula of Z_

    # matrix of the updated cdf's
    FZ_pos_matrix=zeros((k_,j_))
    for k in range(k_):
        FZ_pos_matrix[k]=c[k]*FZ_pos[k](Z_sorted[k])+(1-c[k])*FZ_[k]

    # scenarios of the posterior view variables
    Z_pos=CopMargComb(Z_sorted,FZ_pos_matrix,U_pri)

    # augmentation of the pick matrix
    _,s2 = FPmeancov(X_pri,p)
    a = v@s2
    v_ort = nullspace(a)[1].T
    v_tilde = r_[v,  v_ort]

    # augmentation of the view variables
    Z_tilde_pri = v_tilde@X_pri

    # posterior view variables
    Z_tilde_pos = r_[Z_pos,Z_tilde_pri[k_:,:]]

    # posterior market variables
    X_pos = solve(v_tilde,Z_tilde_pos)
    return X_pos, Z_pri, U_pri, Z_pos, v_tilde, Z_tilde_pri, Z_tilde_pos

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from PrincCompFP import PrincCompFP
from OrdLeastSquareFPNReg import OrdLeastSquareFPNReg
from KalmanFilter import KalmanFilter


def DynPC(X, p, sig2, k_, option=None):
    # This function calibrates the parameters of the Kalman filter model by
    # using a two-step approach, statistical and regression LFM's
    #INPUTS
    # X        :[matrix](n_ x t_end) dataset of observations
    # p        :[vector](1 x t_end) flexible probabilities
    # sig2     :[matrix](n_ x n_) metric matrix
    # k_       :[scalar] number of factors recovered via PC LFM
    # option   :[struct] structure specifying the initial values (needed for extraction) of hidden factors Z_KF and their uncertainty.
    #                    In particular:
    #                    option.Z   :[vector](k_ x 1) initial value of hidden factor
    #                    option.p2  :[matrix](k_ x k_) uncertainty on the initial value Z of hidden factors
    #OPS
    # Z_KF     :[matrix](k_ x t_end) dataset of extracted factors
    # alpha    :[vector](n_ x 1) shifting term in the observation equation
    # beta     :[matrix](n_ x k_) parameter of the observation equation
    # s2       :[matrix](n_ x n_) covariance of invariants in the observation equation
    # alpha_z  :[vector](k_ x 1) shifting term in the transition equation
    # beta_z   :[matrix](k_ x k_) parameter of the transition equation
    # s2_z     :[matrix](k_ x k_) covariance of the invariants in the transition equation

    # For details on the exercise, see here .

    # Step 1
    # Estimation of statistical LFM
    alpha_PC, beta_PC, gamma_PC, s2_U_PC = PrincCompFP(X, p, sig2, k_)
    # set outputs
    alpha = alpha_PC
    beta = beta_PC
    s2 = s2_U_PC
    # compute hidden factors
    Z = gamma_PC@X

    # Step 2
    # Estimation of regression LFM
    alpha_OLSFP, beta_OLSFP, s2_OLSFP, _ = OrdLeastSquareFPNReg(Z[:,1:],Z[:,:-1],p[[0],:-1])
    # set outputs
    alpha_z = alpha_OLSFP
    beta_z = beta_OLSFP
    s2_z = s2_OLSFP

    # Step 3
    # Extraction of hidden factors
    Z_KF = KalmanFilter(X,alpha,beta,s2,alpha_z,beta_z,s2_z,option)
    return Z_KF, alpha, beta, s2, alpha_z, beta_z, s2_z


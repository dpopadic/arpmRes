from numpy import zeros, diag, eye, exp, sqrt

from scipy.linalg import kron


def PortfolioMomentsLogN(v_tnow, h, LogN_loc, LogN_disp):
    # This function computes the expectation, the standard deviation and the skewness
    # of a portfolio's P&L which is lognormally distributed.
    #  INPUTS
    #  v_tnow        : [vector] (n_ x 1)  current values of the instruments
    #  h             : [vector] (n_ x 1)  portfolio's holdings
    #  LogN_loc      : [vector] (n_ x 1)  location parameter of the instruments lognormal distribution at the horizon
    #  LogN_disp     : [vector] (n_ x n_) dispersion parameter of the instruments lognormal distribution at the horizon
    #  OPS
    #  muPL_h      : [scalar] expectation of the portfolio's P&L
    #  sdPL_h      : [scalar] standard deviation of the portfolio's P&L
    #  skPL_h      : [scalar] skewness of the portfolio's P&L

    # portfolio's P&L expectation
    muV_thor = exp( LogN_loc + 1/2*diag(LogN_disp).reshape(-1,1)) # expectation of the portfolio's instruments at the horizon
    muPL = muV_thor - v_tnow # expectation of one unit of the portfolio's P&L
    muPL_h = h.T@muPL # portfolio's P&L expectation

    n_ = v_tnow.shape[0]
    d = eye(n_) # canonical basis
    noncent2ndPL = zeros((n_,n_))
    noncent3rdPL = zeros((n_*n_*n_,1))
    cent2ndPL = zeros((n_,n_))
    cent3rdPL = zeros((n_*n_*n_,1))
    i = 0

    for n in range(n_):
        for m in range(n_):
            omega_nm = d[:,n] + d[:,m]
            # second non-central moments
            noncent2ndV_thor_nm = exp(omega_nm.T@LogN_loc + 1/2*omega_nm.T@LogN_disp@omega_nm)
            noncent2ndPL[n,m] = noncent2ndV_thor_nm - muV_thor[n]*v_tnow[m] - muV_thor[m]*v_tnow[n] + v_tnow[m]*v_tnow[n]
            # second central moment
            cent2ndPL[n,m] = noncent2ndPL[n,m] - muPL[n]*muPL[m]
            for l in range(n_):

                omega_nl = d[:,n] + d[:,l]
                omega_ml = d[:,m] + d[:,l]
                omega_nml = d[:,n] + d[:,m] + d[:,l]
                # second non-central moments that enter in the third non-central moments formulas
                noncent2ndV_thor_nl = exp(omega_nl.T@LogN_loc + 1/2*omega_nl.T@LogN_disp@omega_nl)
                noncent2ndV_thor_ml = exp(omega_ml.T@LogN_loc + 1/2*omega_ml.T@LogN_disp@omega_ml)
                # third non-central moments
                noncent3rdV_thor_nml = exp(omega_nml.T@LogN_loc + 1/2*omega_nml.T@LogN_disp@omega_nml)
                noncent3rdPL[i] = noncent3rdV_thor_nml - noncent2ndV_thor_nm*v_tnow[l] - noncent2ndV_thor_nl*v_tnow[m]+\
                                  muV_thor[n]*v_tnow[m]*v_tnow[l] - noncent2ndV_thor_ml*v_tnow[n] +\
                                  muV_thor[m]*v_tnow[n]*v_tnow[l] + muV_thor[l]*v_tnow[n]*v_tnow[m] -\
                                  v_tnow[m]*v_tnow[n]*v_tnow[l]
                # third central moment
                cent3rdPL[i] = noncent3rdPL[i] - noncent2ndPL[n,m]*muPL[l] - noncent2ndPL[n,l]*muPL[m] -\
                               muPL[n]*noncent2ndPL[m,l] + 2*muPL[n]*muPL[m]*muPL[l]
                i = i + 1

    sdPL_h = sqrt(h.T@cent2ndPL@h) # portfolio's P&L standard deviation
    dummy = kron(h,h)@h.T
    vec_h = dummy.flatten()
    skPL_h = (vec_h.T@cent3rdPL)/(sdPL_h**3) # portfolio's P&L skewness

    return muPL_h, sdPL_h, skPL_h

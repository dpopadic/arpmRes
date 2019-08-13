from collections import namedtuple

import numpy as np
from numpy import roots, real, sign, log, sqrt

from scipy.optimize import minimize


from ShiftedVGMoments import ShiftedVGMoments


def MMFP(HFP, Model, par0=None):
    # This function estimates the parameters of the selected distribution by
    # using the Method of Moments with Flexible Probabilities
    #  INPUTS
    # HFP        :[struct] with fields
    #             - Scenarios [vector](1 x t_end) historical scenarios
    #             - FlexProbs [vector](1 x t_end) Flexible Probabilities associated to historical scenarios
    # Model      :[string] SLN for Shifted LogNormal model
    #                      SVG for Shifted Variance Gamma
    # par0       :[vector](1 x 4) starting guess for SVG parameters use only in the SVG case
    #  OPS
    # Parameters :[struct] with fields
    #             - {sig2, mu, c} in the SLN case
    #             - {c, theta, sigma, nu} in the SVG case

    ## Code

    p=HFP.FlexProbs
    epsi=HFP.Scenarios

    if Model =='SLN':
        m1=p@epsi.T
        m2=p@((epsi-m1)**2).T
        m3=p@((epsi-m1)**3).T

        s=sqrt(m2)
        alpha3=m3/(m2)**(3/2)
        t=[1, 3, 0, -((alpha3)**2+4)]
        w=roots(t)
        wstar = w[w == real(w)]

        Parameters = namedtuple('param',['sig2','mu','c'])
        Parameters.sig2=log(wstar)
        Parameters.mu=log(s*(wstar*(wstar-1))**(-0.5))
        Parameters.c=sign((alpha3))*m1-s*((wstar-1)**(-0.5))

        #shifted Variance-Gamma model
    elif Model == 'SVG':
        m1=p@epsi.T
        m2=p@((epsi-m1)**2).T
        m3=p@((epsi-m1)**3).T
        m4=p@((epsi-m1)**4).T

        s_HFP = sqrt(m2)
        skew_HFP = m3/(m2)**(3/2)
        kurt_HFP = m4/(m2)**2
        lb = [-np.inf, -np.inf, 1e-9, 1e-9]
        ub = [np.inf, np.inf,np.inf,np.inf]
        bounds = list(zip(lb,ub))
        options = {'disp':False,'ftol':1e-15,'gtol':1e-15}
        parameters = minimize(WeightedSquareSum,par0,args=(m1, s_HFP, skew_HFP, kurt_HFP),bounds=bounds,options=options)
        parameters = parameters.x
        Parameters = namedtuple('param',['theta','sigma', 'nu','c'])
        Parameters.c=parameters[0]
        Parameters.theta=parameters[1]
        Parameters.sigma=parameters[2]
        Parameters.nu=parameters[3]
    return Parameters


def WeightedSquareSum(par, m1, s_HFP, skew_HFP, kurt_HFP):
    c = par[0]
    theta = par[1]
    sigma = par[2]
    nu = par[3]
    mu, sig2, skew, kurt = ShiftedVGMoments(c,theta, sigma, nu, 1)
    S = (mu-m1)**2 + (sqrt(sig2)-s_HFP)**2 + (skew-skew_HFP)**2 + (kurt-kurt_HFP)**2
    return S.squeeze()

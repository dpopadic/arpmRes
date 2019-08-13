from collections import namedtuple

import numpy as np
from numpy import ones, eye, abs, mean, sqrt, r_
from numpy.linalg import solve

from scipy.optimize import minimize


def IterGenMetMomFP(epsi,p,Model,Order=2):
    # This function computes the generalized method of moments with flexible
    # probabilities estimate for the parameter lam of an invariant which is
    # Poisson distributed
    # INPUTS:
    #   epsi              : [vector] (1 x t_end) time series of the invariant
    #   p                 : [vector] (1 x t_end) Flexible Probabilities
    #   Model:            : [string] Poisson for Poisson distribution
    #   Order:            : [scalar] unused input for Poisson distribution
    # OUTPUTS:
    #   Parameters         : [struct] with fields
    #   Parameters.lam  : [scalar] GMMFP estimate of the parameter of the Poisson distribution

    # For details on the exercise, see here .

    ## Code

    if Model=='Poisson':

        # general settings
        NmaxIter=100
        lam=mean(epsi)*ones(NmaxIter)
        conv=0
        i=1
        t_ = p.shape[1]
        # 0. Set initial weighting matrix omega_2
        omega_2=eye(2)

        #Set initial vector v_lamda and initial quadratic form in omega_2
        a=(epsi-lam[0])
        b=(epsi**2)-lam[0]*(lam[0]+1)
        v_lamda=r_[p@a.T, p@b.T]
        quadform=v_lamda.T@omega_2@v_lamda@ones((1,NmaxIter))

        while i<NmaxIter and conv==0:

            # 1. Update output lam
            lam[i] = GMMpoisson(epsi,p,omega_2,lam[i-1])#compute the new lam

            #2. Update weighting matrix omega_2
            a=(epsi-lam[i])
            b=(epsi**2)-lam[i]*(lam[i]+1)
            v_lamda=r_[p@a.T, p@b.T]
            rhs = r_[r_['-1', p@(a**2).T, p@(a*b).T], r_['-1', p@(a * b).T, p@(b ** 2).T]]
            omega_2=solve(rhs,eye(rhs.shape[0]))

            #shrinkage towards identity of weighting matrix omega_2
            aa=sqrt(p@(a**2).T)
            bb=sqrt(p@(b**2).T)
            c=(omega_2/r_[r_['-1',aa**2, aa*bb], r_['-1',aa*bb, bb**2]])
            omega_2=0.5*np.diagflat(r_['-1',aa, bb])@c@np.diagflat(r_['-1',aa, bb])+0.5*eye(2) #new weighting matrix

            # 3. If convergence, return the output, else: go to 1
            quadform[0,i]=v_lamda.T@omega_2@v_lamda
            reldistance=abs((quadform[0,i]-quadform[0,i-1])/quadform[0,i-1])
            if reldistance < 10**-8:
                conv=1
            i=i+1

        Parameters = namedtuple('lam',['lam'])
        Parameters.lam=lam[i-1]

    return Parameters


def GMMpoisson(epsi,p,omega2,lambda0):
    # This function solves the minimization problem in lam argmin v_lamda.T@omega2_lamda@v_lamda
    # (it is called by the function IterGenMetMomFP in case of Poisson distribution)
    # INPUTS
    #  epsi    :[vector] (1 x t_end) time series of the invariant
    #  p       :[vector] (1 x t_end) Flexible Probabilities
    #  omega2  :[matrix] (2 x 2)  weighting matrix
    #  lambda0 :[scalar] initial value of the minimization algorithm
    # OP
    #  lam  :[scalar] new output for function IterGenMetMomFP

    # Optimization options
    # options.TolFun = 1e-8
    # options.MaxFunEvals =5000
    # options.MaxIter	= 5000
    # options.Display	=off
    # options.TolX=1e-8

    options = {'maxiter' : 5000}

    # Solve the minimization problem
    lam = minimize(GMMp,lambda0,args=(omega2,epsi,p),options=options, tol=1e-8)
    return lam.x


def GMMp(lambda0,omega2,epsi,p):
    a=(epsi-lambda0)
    b=(epsi**2)-lambda0*(lambda0+1)

    v=r_[p@a.T, p@b.T]
    F=v.T@omega2@v
    return F.squeeze()

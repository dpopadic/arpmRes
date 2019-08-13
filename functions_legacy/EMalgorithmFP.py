import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, isnan, ones, zeros, tile, r_, diagflat
from numpy import sum as npsum
from numpy.linalg import norm, inv

plt.style.use('seaborn')

from FPmeancov import FPmeancov


def EMalgorithmFP(epsi,FP,nu,tol):
    #Expectation-Maximization with Flexible Probabilities for Missing Values
    #under Student t assumption (nu degrees of freedom)
    # INPUT
    # epsi         : [matrix] (i_ x t_end) observations - with zeros's for missing values
    # FP           : [vector] (1 x t_end) flexible probabilities
    # nu           : [scalar] multivariate Student's t degrees of freedom
    # tol          : [scalar] or [vector] (2 x 1) tolerance, needed to check convergence of mu and sigma2 estimates
    # OP
    # mu           : [vector] (i_ x 1)  EMFP estimate of the location parameter
    # sigma2       : [matrix] (i_ x i_) EMFP estimate of the scatter matrix

    # For details on the exercise, see here .

    #tolerance: needed to check convergence
    if isinstance(tol, float) or len(tol)==1:
        tol=[tol, tol]

    i_,t_=epsi.shape

    #step0: initialize
    I = isnan(epsi)

    Data = epsi[:,npsum(I,axis=0)==0]

    FPa = FP[[0],npsum(I,axis=0)==0].reshape(1,-1)
    FPa=FPa/npsum(FPa)

    # HFP mu and sigma2 on available data
    m, s2=FPmeancov(Data,FPa)
    # m = m[...,np.newaxis]
    s2 = s2[...,np.newaxis]

    w=ones((1,t_))

    Error=ones(len(tol))*10**6
    j=0
    # start main loop
    gamma = {}
    while any(Error>tol):
        j=j+1
        eps = zeros((epsi.shape[0],t_))
        for t in range(t_):
            gamma[t]=zeros((i_,i_))

            na=[]
            for i in range(i_):
                if isnan(epsi[i,t]):
                    na=r_[na, i] #non-available

            a= arange(i_)
            if isinstance(na,np.ndarray):
                if na.size > 0:
                    mask = np.ones(a.shape, dtype=bool)  # np.ones_like(a,dtype=bool)
                    na = list(map(int,na))
                    mask[na] = False
                    a = a[mask] #available

            A=i_-len(na) #|available|

            eps[a,t]=epsi[a,t]
            eps[na,t]=epsi[na,t]

            #step1:

            #update weights
            invs2 = inv(s2[np.ix_(a,a,[j-1])].squeeze())
            w[0,t]=(nu+A)/(nu+(eps[a,[t]]-m[a,[j-1]]).T@invs2@(eps[a,[t]]-m[a,[j-1]]))

            if na:
                #fill entries
                eps[na,t]=(m[na,[j-1]]+s2[np.ix_(na,a,[j-1])].squeeze()@invs2@(eps[a,[t]]-m[a,[j-1]])).flatten()

                #fill buffer
                gamma[t][np.ix_(na,na)]=s2[np.ix_(na,na,[j-1])].squeeze()-s2[np.ix_(na,a,[j-1])].squeeze()@invs2@s2[np.ix_(a,na,[j-1])].squeeze()

        #step[1:] update output
        new_m=(eps@(FP*w).T)/npsum(FP*w)
        m = r_['-1',m,new_m]
        gamma_p = zeros(gamma[0].shape+(t_,))
        for t in range(t_):
            gamma_p[:, :, t]= gamma[t]*FP[0,t]
        new_s2= (eps-tile(m[:,[j]],(1,t_)))@(diagflat(FP*w))@(eps-tile(m[:,[j]],(1,t_))).T+npsum(gamma_p,2)
        s2 = r_['-1',s2,new_s2[...,np.newaxis]]

        # step3: check convergence
        Error[0] = norm(m[:,j]-m[:,j-1])/norm(m[:,j-1])
        Error[1] = norm(s2[:,:,j]-s2[:,:,j-1],ord='fro')/norm(s2[:,:,j-1],ord='fro')

        mu=m[:,-1]
        sigma2=s2[:,:,-1]
    return mu, sigma2

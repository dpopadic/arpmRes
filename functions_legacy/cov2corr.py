from numpy import diag, sqrt, zeros


def cov2corr(cov):
    # Convert covariance matrices to correlation matrices.
    #
    # USAGE:
    #  [SIGMA,CORREL] = cov2corr(COV)
    #
    # INPUTS:
    #   COV   - A K by K covariance matrix -OR-
    #           A K by K by T array of covariance matrices
    #
    # OUTPUTS:
    #   SIGMA  - A K by 1 vector of standard deviations if COV is K by K -OR-
    #            A T by K matrix of standard deviations
    #   CORREL - A K by K matrix of correlations -OR-
    #            A K by K by T matrix of correlations.
    #
    # EXAMPLES:
    #   # DCC(1,1)
    #   [~,~,Ht] = dcc(data,[],1,0,1)
    #   [S,Rt] = cov2corr(Rt);
    #
    # See also DCC
    # For details, see here.

    # Copyright: Kevin Sheppard
    # kevin.sheppard@economics.ox.ac.uk
    # Revision: 1    Date: 10/23/2012

    if cov.ndim==2:
        sigma = sqrt(diag(cov)).reshape(-1,1)
        correl = cov/(sigma@sigma.T)
    elif cov.ndim==3:
        T = cov.shape[2]
        K = cov.shape[1]
        sigma = zeros((T,K))
        correl = zeros((K,K,T))
        for t in range(T):
            sigma[t,:] = sqrt(diag(cov[:,:,t]))
            correl[:,:,t] = cov[:,:,t]/(sigma[[t],:].T@sigma[[t],:])
    return sigma, correl

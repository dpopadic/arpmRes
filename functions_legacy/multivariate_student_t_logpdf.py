import numpy as np
from scipy.special import gamma
from numpy.linalg import slogdet

def multivariate_student_t_logpdf(X, mu, Sigma, df):
    #multivariate student T log pdf
    n = X.shape[0]
    Xm = X-mu
    V = df * Sigma
    V_inv = np.linalg.inv(V)
    (sign, logdet) = slogdet(np.pi * V)

    logz = -gamma(df/2.0 + n/2.0) + gamma(df/2.0) + 0.5*logdet
    logp = -0.5*(df+n)*np.log(1+ np.sum(np.dot(V_inv,Xm)*Xm,axis=0))

    logp = logp - logz

    return logp
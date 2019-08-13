import matplotlib.pyplot as plt
from numpy import sum as npsum
from numpy import zeros, linspace, cov, mean, exp
from numpy.linalg import solve

plt.style.use('seaborn')

from EffectiveScenarios import EffectiveScenarios
from SpinOutlier import SpinOutlier


def BlowSpinFP(z,b_,s_,blowscale=[.1, 3],spinscale=1,method='inverse'):
    # This function generates a pool of Flexible Probabilitiy profiles for the
    # dataset z based on a bivariate Gaussian kernel.
    #  INPUTS
    #  z          :[matrix] (2 x t_end) dataset
    #  b_         :[scalar] number of blow frames (kernels with fixed center, "increasing" dispersion)
    #  s_         :[scalar] number of spin frames (kernels with fixed dispersion matrix, centers arranged around a circle)
    #  blowscale  :[vector] (1 x 2) min and max scale for the blow frames
    #  spinscale  :[scalar] scale for the spin frames
    #  method     :[string] direct: FP are proportional to the normal PDF
    #                       inverse(default): FP are proportional to (1- normal PDF)
    #  OPS
    #  p          :[matrix] ((b_+s_) x t_end) Flexible Probability profiles
    #  ens        :[vector] (1 x (b_+s_)) effective number of scenarios for each profile

    # For details on the exercise, see here .
    ## Code

    q_ = b_+s_ #total number of FP profiles
    t_ = z.shape[1] #number of observations
    mu = mean(z,1,keepdims=True)
    sigma2 = cov(z)

    #initialization
    ens = zeros((1,q_)) #effective number of scenarios
    p = zeros((q_,t_)) #matrix containing the q_ FP-profiles

    #Blow
    if b_ ==1:
        b_scale = [blowscale[1]]
    else:
        b_scale=linspace(blowscale[0],blowscale[1],b_)

    for b in range(b_):
        for t in range(t_):
             z2 = b_scale[b]*(z[:,[t]]-mu).T@solve(sigma2,z[:,[t]]-mu)
             if method == 'direct':
                 p[b,t] = exp(-z2)
             else:
                 p[b,t] = 1-exp(-z2)
        p[b,:] = p[b,:]/npsum(p[b,:])
        ens[0,b] = EffectiveScenarios(p[b,:])

    #Spin
    if s_>0:
        m=SpinOutlier(mu,sigma2,spinscale,s_)
    else:
        m=[]

    for q in range(b_,q_):
        for t in range(t_):
             z2 = (z[:,t]-m[:,q-b_]).T@solve(sigma2,z[:,t]-m[:,q-b_])
             p[q,t] = exp(-z2)
        p[q,:] = p[q,:]/sum(p[q,:])
        ens[0,q] = EffectiveScenarios(p[q,:])

    return p,ens

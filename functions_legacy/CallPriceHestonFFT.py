import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, pi, real, interp, floor, log, exp, sqrt
from numpy.fft import fft
from scipy.interpolate import interp1d

plt.style.use('seaborn')


def CallPriceHestonFFT(s_0,k,r,tau,z):
    # This function computes the Heston price of european call options using FFT
    # INPUTS
    # s_0  :[scalar] current price of the underlying
    # k    :[row vector] strike values
    # r    :[scalar] risk free rate
    # tau  :[scalar] time to maturity
    # z    :[vector] (1 x 5) Heston model parameters
    # OUTPUTS
    # c_heston  :[row vector] Heston prices for european call options

    ## Code

    n_FFT=1024*2
    eta=0.250
    alpha=1.5# dampening factor

    lam=(2*pi)/(n_FFT*eta)
    b=0.5*n_FFT*lam

    logstrike=-b+lam*arange(0,n_FFT-1+1,1).T
    min_k=int(-(floor((-log(0.65)/lam)+1))+n_FFT/2+2)-1
    max_k=int(floor((log(1.45)/lam))+1+n_FFT/2+1)
    strikeFFT=s_0*exp(logstrike[min_k:max_k])
    v=eta*arange(0,n_FFT-1+1,1).T
    simpson_weights=simpson(n_FFT)

    kernel_points=kernel(v,alpha,r,0,z,tau,log(s_0))
    coeff=exp(1j*b*v)*kernel_points*simpson_weights
    price_tmp=fft(coeff)
    price_tmp1=real(price_tmp)*(exp(-alpha*logstrike)/pi)*eta
    price_tmp2=price_tmp1[min_k:max_k]
    interp = interp1d(strikeFFT,price_tmp2, fill_value='extrapolate')
    c_heston=interp(k)
    return c_heston


def simpson(n):
    simpson_weights=(1/3)*(3+(-1)**(arange(1,n+1,1).T))
    simpson_weights[0]=1/3
    return simpson_weights


def kernel(v,alpha,y,d,p,t,s):
    CF_points=CF(v-1j*(1+alpha),y,d,p,t,1)
    kernel_points=exp(-y*t+s)*CF_points/(alpha**2+alpha-v**2+1j*(2*alpha+1)*v)
    return kernel_points


def CF(v,y,d,p,t,n):
    # kappa = 1.49 #mean reverting velocity
    # theta = 0.0671 #mean reverting
    # sigma = 0.742 #vol_vol
    # rho = -0.571#correlation
    # sigma0 = 0.0262#initial variance
    # lp = [k,theta,sigma,rho,sigma0].T
    t=t/n
    kappa=p[0]
    theta=p[1]
    sigma=p[2]
    rho=p[3]
    sigma0=p[4]
    A=1j*v*((y-d)*t)
    zeta_=-0.5*(v**2+1j*v)
    gamma_=kappa-rho*sigma*v*1j
    psi_=sqrt(gamma_**2-2*sigma**2*zeta_)
    B=2*zeta_*(1-exp(-psi_*t))*sigma0/(2*psi_-(psi_-gamma_)*(1-exp(-psi_*t)))
    C=-(kappa*theta/sigma**2)*(2*log((2*psi_-(psi_-gamma_)*(1-exp(-psi_*t)))/(2*psi_))+(psi_-gamma_)*t)
    CF_points=exp(A+B+C)
    return CF_points

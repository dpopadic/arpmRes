from collections import namedtuple

import matplotlib.pyplot as plt
from numpy import ones, zeros, log, exp, tile, r_
from numpy import sum as npsum

from L1GeneralGrafting import L1GeneralGrafting



def features_form(X, z, method):
    if method =='linear': # linear features
        y = (tile(X, (1, z.shape[1])) - 1/2)*z
    elif method == 'pure quadratic': # quadratic features
        y_quad = zeros((z.shape-1))
        for t in range(z.shape[0]):
            y_quad[t] = ones((1, z.shape[1]))@(z[t, 1:].T@z[t, 1:])*(X[t] - 1/2)
        y = r_[(X - 1/2)*z[:,0], y_quad]
    elif method == 'quadratic': # linear and quadratic features
        y_lin = (tile(X, (1, z.shape[1])) - 1/2)*z
        y_quad = zeros((z.shape-1))
        for t in range(z.shape[0]):
            y_quad[t] = ones((1, z.shape[1]-1))@(z[t, 1:].T@z[t, 1:])*(X[t] - 1/2)
        y = [y_lin,y_quad]
    return y


def LogLik(theta, z, X, method, n_):
    # negative log-likelihood function
    negloglik = - 1/n_*sum(-log(exp(features_form(zeros((n_, 1)), z, method)@theta) +\
        exp((features_form(ones((n_,1)), z, method)@theta))) + sum(features_form(X, z, method)@theta))

    # negative log-likelihood gradient
    gradient = -1/n_*npsum( features_form(X, z, method)
                            - 1/tile(exp(features_form((zeros(n_,1)), z, method)@theta) +
                                     exp(features_form(ones((n_,1)), z, method)@theta), (1, len(theta)))*\
        (tile(exp(features_form(zeros((n_,1)), z, method)@theta), (1, len(theta)))*
         features_form(zeros((n_,1)), z, method) +tile(exp(features_form(ones((n_,1)), z, method)@theta),(1, len(theta)))*
         features_form(ones((n_,1)), z, method)))
    gradient = gradient.T

    return negloglik, gradient


def FitDefaultProb(x, z, lam, method):
    #Estimation of default probabilities via Maximum Likelihood with lasso
    #penalties (Logit model)
    # INPUT
    # x      :[vector] (n_x 1) default indicator (0/1)
    # z      :[matrix] (n_ x k_)obligor-specific variables
    # lam :[scalar] lasso parameter
    # method :[string] linear, quadratic or .Tpure quadratic.T
    # OP
    # score  :[struct] with fields:
    #                  theta: estimated parameters
    #                  p_z: estimated default probabilities
    #######################################################

    # add bias
    n_, k_ = z.shape
    z = r_['-1',ones((n_,1)), z]
    k_ = k_ + 1

    # loss function
    loss = LogLik
    loss_input = [z, x, method]

    # set lasso penalties
    if method=='linear' or method=='pure quadratic':
        theta = zeros((k_,1))
        lambdaVect = [0, lam*ones((k_-1,1))] # penalties
    elif method=='quadratic':
        theta = zeros((k_*2-1, 1))
        lambdaVect = [0, lam*ones((k_*2-2,1))] # penalties

    options = namedtuple('options',['order', 'verbose','mode'])

    # Use default options
    options.order = 1 # Choose whether to require Hessian information explicitly (change this to 1 to use BFGS versions)
    options.verbose = 0# To get more accurate timing, you can turn off verbosity:
    options.mode = 1

    # MLE via grafting algorithm
    score = namedtuple('score','theta p_z')
    score.theta = L1GeneralGrafting(loss,theta,lambdaVect,options,loss_input)

    # scoring function: default probailities
    for i in range(n_):
        if x[i] == 0:   # no default
            score.p_z[i] = 1 - exp((features_form(x[i], z[i, :], method)@score.theta)/
                (exp(features_form(1, z[i, :], method)@score.theta) + exp(features_form(0, z[i, :], method)@score.theta)))
        else:
            score.p_z[i] = exp((features_form(x[i], z[i, :], method)@score.theta) / ( exp(features_form(1, z[i, :], method)@score.theta)
                                                                                      + exp(features_form(0, z[i, :], method)@score.theta)))

    score.p_z = score.p_z.T
    return score


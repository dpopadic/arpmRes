from numpy import trace, abs, exp


def ObjectiveToeplitz(gamma,tau_x,tau_y,c2):
    # Computes the square Frobenius norm of the difference between the empirical and
    # theoretical correlation matrices
    #  INPUTS
    # gamma  :[scalar] parameter of the theoretical correlation
    # tau_x  :[matrix](n_ x n_) matrix of times to maturities, constant in columns
    # tau_y  :[matrix](n_ x n_) matrix of times to maturities, constant in rows
    # c2     :[matrix](n_ x n_) empirical correlation matrix
    #  OP
    # error  :[scalar] square Frobenius norm of the difference between the correlation matrices

    error = trace((exp(-gamma*abs(tau_x-tau_y))-c2)@(exp(-gamma*abs(tau_x-tau_y))-c2).T)
    return error

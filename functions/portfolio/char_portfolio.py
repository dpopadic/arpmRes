import numpy as np
from scipy.optimize import minimize


def char_portfolio(beta, sig2_pl, h_0=None, h_tilde=None,
                   sigma2_max=None, v=None, v_0=None):
    """For details, see here.

    Parameters
    ----------
        beta : array, shape(n_, )
        sig2_pl : array, shape(n_, n_)
        k_ : int
        h_0: array, optional, shape(n_, )
        h_tilde: array, optional, shape(n_, )
        sigma_2_max: scalar, optional
        v: array, shape(n_, )
        v_0: scalar

    Returns
    -------
        h_char : array, shape(n_, )

    """

    beta = beta.reshape(-1, 1)
    if (v is None and h_tilde is None and h_0 is None and
            v_0 is None and sigma2_max is None):

        h_char = np.linalg.solve(sig2_pl, beta) / \
            (beta.T @ np.linalg.solve(sig2_pl, beta))

    else:
        # zero correlation constraint
        if h_tilde is not None:
            aeq = h_tilde.reshape(1, -1)@sig2_pl
            beq = np.array([[0]])

        # budget constraint
        if v_0 is not None:
            aeq = np.r_[aeq, v.reshape(1, -1)]
            beq = np.r_[beq, np.atleast_2d(v_0)]

        eqcon = lambda x, A, b: -((A @ x)-b).flatten()

        opts = {'maxiter': 10**6, 'disp': False}

        # upper bound constraint on variance
        cons = (
            {'type': 'ineq', 'fun': lambda h, sig2_pl,
             sigma2_up: -(h.T@sig2_pl@h-sigma2_up),
             'args': (sig2_pl, sigma2_max)},
            {'type': 'ineq', 'fun': lambda h,
             sig2_pl: -(-h.T@sig2_pl@h + 10**-6), 'args': (sig2_pl, )},
            {'type': 'eq', 'fun': eqcon, 'args': (aeq[0], beq[0])},
            {'type': 'eq', 'fun': eqcon, 'args': (aeq[1], beq[1])}
        )
        objfun = lambda h, beta: (-h @ beta)
        res = minimize(objfun, h_0, method='SLSQP', args=(beta,),
                       jac=False, constraints=cons, options=opts)
        h_char = res.x
    return h_char.flatten()

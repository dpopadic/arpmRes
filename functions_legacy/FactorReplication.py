import matplotlib.pyplot as plt
from numpy import r_, array, atleast_2d
from numpy.linalg import solve
from scipy.optimize import minimize

plt.style.use('seaborn')


def FactorReplication(beta, sigma2_pnl, constraints=None):
    # This function computes the factor replicating portfolios based on
    # characteristic portfolios
    #  INPUTS
    #   beta        : [vector] (n_ x 1)  characteristics
    #   sigma2_pnl  : [matrix] (n_ x n_) conditional P&L covariance matrix
    #   constraints : [structure]        portfolio constraints inputs
    #  OPS
    #   h           : [vector] (n_ x 1)  characteristic portfolio

    # For details on the exercise, see here .
    ## Code

    if constraints is None:
        h = solve(sigma2_pnl,beta)/(beta.T@solve(sigma2_pnl,beta)) # characteristic portfolio
    else:
        # linear inequality constraints

        if 'v_up' in constraints._fields:
            a = constraints.v.T
            b = constraints.v_up # upper bound on budget
        if 'v_down' in constraints._fields:
            a = r_[a, -constraints.v.T]
            b = r_[b, -constraints.v_down] # lower bound on budget

        # linear equality constraints
        if 'h_tilde' in constraints._fields:
            aeq = constraints.h_tilde.reshape(1,-1)@sigma2_pnl # no correlation with h_tilde
            beq = array([[0]])

        if 'v_0' in constraints._fields:
            aeq = r_[aeq, constraints.v.T] # no corr + full investment constr
            beq = r_[beq, constraints.v_0]
        if 'dollar_neutral' in constraints._fields:
            aeq = r_[aeq, constraints.v.reshape(1,-1)] # no dollar neutral constr
            beq = r_[beq, atleast_2d(constraints.dollar_neutral)] # constraints.dollar_neutral must be equal 0

        # nonlinear constraints
        # nonlcon = lambda x, arg1, arg2 : nonlinconstr(x, arg1,arg2)
        # nonlconjac = lambda x, arg1, arg2 : nonlinconstrjac(x, arg1,arg2)
        # ineqcon = lambda x, A, b: -A.dot(x)+b
        eqcon = lambda x, A, b: -(A.dot(x)-b).flatten()
        # flexible characeristic portfolio
        opts = {'maxiter': 10**6, 'disp':False}
        cons = (
            {'type': 'ineq', 'fun': lambda x, sigma2_pnl,sigma2_up: -(x.T@sigma2_pnl@x-sigma2_up), 'args':(sigma2_pnl,constraints.sigma2_max)},
            {'type': 'ineq', 'fun': lambda x, sigma2_pnl: -(-x.T@sigma2_pnl@x + 10**-6), 'args':(sigma2_pnl, )},
            {'type': 'eq', 'fun': eqcon, 'args':(aeq[0], beq[0])},
            {'type': 'eq', 'fun': eqcon, 'args':(aeq[1], beq[1])}
        )
        objfun = lambda x, beta: (-x.flatten().dot(beta))
        res = minimize(objfun, constraints.h_0,method='SLSQP', args=(beta,), jac=False, constraints=cons, options=opts)
        h = res.x
    return h

# nonlinear constraints function
def nonlinconstr(x, sigma2_pnl, sigma2_up):
    # This function is used to impose nonlinear constraints on flexible
    # characteristic portfolio
    #  INPUTS
    #   x       : [vector] (n_ x 1)  unknown
    #   sigma2  : [matrix] (n_ x n_) P&L variance
    #   s2_up   : [scalar]           upper bound portfolio variance
    #  OPS
    #   c       : [scalar]           inequality constraint
    #   ceq     : [scalar]           equality constraint
    #   gradc   : [vector] (n_ x 1)  inequality constraint gradient
    #   gradceq : [vector] (n_ x 1)  equality constraint gradient

    ## Code
    c = array([x.T@sigma2_pnl@x - sigma2_up, -x.T@sigma2_pnl@x + 10**-6]) # lower and upper bound for conditional variance
    # ceq = []
    # gradc = array([2*sigma2_pnl@x, -2*sigma2_pnl@x])
    # gradceq = []
    return c

def nonlinconstrjac(x, sigma2_pnl, sigma2_up):
    # This function is used to impose nonlinear constraints on flexible
    # characteristic portfolio
    #  INPUTS
    #   x       : [vector] (n_ x 1)  unknown
    #   sigma2  : [matrix] (n_ x n_) P&L variance
    #   s2_up   : [scalar]           upper bound portfolio variance
    #  OPS
    #   gradc   : [vector] (n_ x 1)  inequality constraint gradient

    ## Code
    gradc = array([2*sigma2_pnl@x, -2*sigma2_pnl@x])
    return gradc
from numpy import array, zeros, eye
from numpy.linalg import solve, pinv


def SolveGarleanuPedersen(n,s,k,epsilon,y,lam,omega,phi2,Phi2,delta,b,h_0):
    # This function computes the numerical solution of the Garleanu-Pedersen
    # model by first solving the Riccati ODE (a_1) and the first order ODE
    # involving a_2.
    # INPUTS
    # n           :[scalar] number of discretization intervals
    # s           :[scalar] number of traded assets
    # k           :[scalar] number of predicting factors
    # epsilon     :[scalar] discretization increment used in finite differences
    #              approximation
    # y           :[scalar] interest rate
    # lam      :[scalar] risk aversion coefficient
    # omega       :[matrix] (s x s) price variance matrix multiplied by its
    #              transpose
    # phi2        :[matrix] (s x s) matrix appearing in the linear market impact
    # Phi2        :[matrix] (k x k) factors mean reversion matrix
    # delta       :[vector] (k x 1) factors variance
    # b           :[matrix] (s x k) component of the alpha term (together with the trading factors f)
    # h_0         :[vector] (1 x s) initial holdings
    # OP
    # h           :[matrix] ((s x (n+1)) discretized optimal trading trajectory in
    #                       the Garleanu-Pedersen model

    # solution of the Riccati ODE (a_1) and solution of the first order ODE (a_2)
    z = zeros((2*s,s,n+1))
    a1 = zeros((s,s,n+1))
    a2 = zeros((s,k,n+1))
    # conditions at time 0
    z[:s,:,n] = array([[1.1, 5.2, 2], [2.3, 4.2, 1], [5.1, 1, 5.18]]) # it can be any value
    z[s:2*s,:,n] = 0
    a1[0,:,n] = 0
    a2[:,:,n] = 0

    for i in range(1,n+1):
        z[:s,:,n-i] = z[:s,:,n+1-i]-epsilon*z[s:2*s,:,n+1-i]
        z[s:2*s,:,n-i] = z[s:2*s,:,n+1-i]-epsilon*(y*z[s:2*s,:,n+1-i]-lam*(omega.dot(pinv(phi2)))@z[:s,:,n+1-i])
        a1[:,:,n-i] = phi2@z[s:2*s,:,n-i].dot(pinv(z[:s,:,n-i]))
        a2[:,:,n-i] = a2[:,:,n+1-i]-epsilon*(a2[:,:,n+1-i]@Phi2-(a1[:,:,n+1-i].dot(pinv(phi2)))@a2[:,:,n+1-i]-b+y*a2[:,:,n+1-i])

    # predicting factors evolution
    f = zeros((k,1,n+1))
    for i in range(1,n+1):
        f[:,:,i] = -Phi2@f[:,:,i-1]+delta
    # conditions at time 0 of the optimal trading trajectory
    h = zeros((s,n+1))
    h[:,0] = h_0
    inv_phi2 = solve(phi2,eye(phi2.shape[0]))
    for i in range(1,n+1):
        h[:,[i]] = h[:,[i-1]]+epsilon*inv_phi2@(a2[:,:,i-1]@f[:,:,i-1]-a1[:,:,i-1]@h[:,[i-1]])

    return h

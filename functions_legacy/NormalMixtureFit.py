from numpy import zeros, abs, log, exp, sum as npsum, max as npmax, ones, var
from numpy.random import rand

from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans


def NormalMixtureFit(X, num_clusters, debug_mode, show_graph, noisy_init):
    # original function created by
    # Marek Grzes
    # University of Waterloo, 2011
    # ---
    # ??? To be revised to make it compliant with our guidelines
    # Learning Gaussian Mixtures using the EM algorithm.
    #   This function is implemented according to Ch. Bishop's book Pattern
    #   recognition and machine learning. Section 9.2 very well describes
    #   learning GMM with the EM algorithm. Pseudo-code from page 438 is used
    #   in this implementaiton.
    #   Fits a Gaussian mixture distribution with K components to the data in X.
    #   X is an N-by-D matrix. Rows of X correspond to observations columns
    #   correspond to variables.
    #   @param debug_mode 0 - no debugging information printed 1 - print debug
    #   values
    #   @param show_graph 0 - no ploting of the current model 1 - plot current
    #   model after each iteration
    #   @param noisy_init 0 - initialistion from kmeans 1 - initialisation
    #   from kmeans but randomly disturbed in order to have worse initial model
    #   @return gm object is returned (gmdistribution.fit returns an object of the
    #   same class).

    # (@) This is how GMM/EM can be done using Matlab functions
    matlab_obj = 0

    # (@) From here my code

    [num_observations, num_dimensions ] = X.shape
    #minx = min((min(X)))
    #maxx = max(max(X))

    # (@) mu, Sigma, and PComponents are things we will be computing in the
    # maximisation step of the EM algoritm, we are going to select them for the
    # expected posterior probabilities p(j|x) using likelihood maximisation.

    # rows are mixture components and columns featues (as in data)
    # (@) init mu randomly
    #mu = minx + (maxx - minx)@rand(num_clusters, num_dimensions)
    # (@) init mu from kmeans
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    kmeanscids, mu = km.labels_, km.cluster_centers_

    # add some noise for the demonstration
    if noisy_init == 1:
        for k in range(num_clusters):
            mult=1
            if rand(0) < 0.5:
                mult = -1
            for d in range(num_dimensions):
                mu[k,d] = mu[k,d] + mult*3 * rand(0)

    # component number is the last dimention so Sigma((:,:,k)) is the kth
    # covariance matrix.
    Sigma = zeros((num_dimensions, num_dimensions, num_clusters))
    for i in range(num_dimensions):
        for j in range(num_dimensions):
            for k in range(num_clusters):
                if i==j:
                    # I just make it a diagonam matrix, because it has to be
                    # positive semidefinite. When computing variance, I compute
                    # variance for the cluster kth only where clusters are
                    # defined according to kmeanscids found by kmeans above.
                    Sigma[i,j,k] = var(X[kmeanscids==k,i],ddof=1)
                    if noisy_init ==1:
                        # add some noise for demonstration
                        Sigma[i,j,k] = Sigma[i,j,k] +4 * rand[0]

    # prior probability of each component: p([j])~uniform
    PComponents = ones((1,num_clusters))*1/num_clusters

    # (@) so here, we have initial values of mu, Sigma and PComponents
    if debug_mode == 1:
        print('initial mu')
        print(mu)
        print('initial sigma')
        print(Sigma)
        print('inital prior probs of clusters')
        print(PComponents)

    ## --------- THE EM ALGORITHM ---------

    # [0] compute initial value of the log likelihood
    old_log_likelihood = 0
    for n in range(num_observations):
        log_cluster_lp = zeros((1, num_clusters))
        for k in range(num_clusters):
            log_cluster_lp[0,k] = log(mvn.pdf(X[n,:], mu[k,:], Sigma[:,:,k]) )
            log_cluster_lp[0,k] = log_cluster_lp[0,k] + log( PComponents[0, k] )

        # internal sum of log likelihood (adding probabibilites)
        log_sum_cluster_lp = log(npsum(exp(log_cluster_lp),1))
        # external sum of log likelihood (adding logarithms)
        old_log_likelihood = old_log_likelihood + log_sum_cluster_lp
    if debug_mode == 1:
        print('initial log likelihood is:')
        print(old_log_likelihood)

    curr_iter = 1
    while True:

        if (show_graph == 1):
            plot_current_model(X, mu, Sigma, PComponents)

        # [1] E-step: compute responsibilities using the current parameter values,
        # by responsibilities we mean posterior probability that given observation
        # x**n was genertated by cluster j, which is p(j|x**n). We compute this from
        # the Bayes theorem by multiplying prior probability of the cluster p([j]) by
        # the likelihood p(x**n|j) of given x**n.
        log_lp = zeros((num_observations, num_clusters)) # for storing log(p(x|j)*p([j]))
        log_posterior_j_on_x = zeros((num_observations, num_clusters))
        # firstly compute all log(p(x|j)p([j]))
        for n in range(num_observations):
            for k in range(num_clusters):
                log_lp[n,k] = log( mvn.pdf(X[n,:], mu[k,:], Sigma[:,:,k]) )
                log_lp[n,k] = log_lp[n,k] + log( PComponents[0, k] )
        # log_sum_exp returns log(sum(exp([a],dim))) while avoiding numerical underflow:
        # dim=2 will sum over columns (here for each obervation we are summing out
        # clusters).
        log_sum_lp = log(npsum(exp(log_lp),1))
        # now compute log posterior probabilities for each pair (n,k)
        for n in range(num_observations):
            for k in range(num_clusters):
                log_posterior_j_on_x[n, k] = log_lp[n,k] - log_sum_lp[n]

        #print(.Testimated resposibilities.T)
        #print((posterior_j_on_x))

        # [2] M-step: re-estimate the parameters using the current responsibilites,
        # that is, compute mu, Sigma, and PComponents using maximul likelihood
        # estimation equations.

        # (3.1) compute Nk for all data points, sum over rows here in order to get
        # log(Nk) for each cluster (NEHE SECOND PARAMETER IS 1 NOW).
        log_Nk = log(npsum(exp(log_posterior_j_on_x), 0))

        # (3.2) re-estimate mu
        for k in range(num_clusters):
            for d in range(num_dimensions):
                ssum = 0
                for n in range(num_observations):
                    ssum = ssum + exp( log_posterior_j_on_x[n,k] - log_Nk[k] )*X[n,d]
                mu[k,d] = ssum

        if debug_mode == 1:
            print('mu')
            print(mu)

        # (3.3) re-estimate Sigma
        for k in range(num_clusters):
            ssum = 0
            for n in range(num_observations):
                # when computing covariance vertical vector@horizontal verctor
                # (see covariance.m in examples).
                ssum = ssum + exp(log_posterior_j_on_x[n,k] - log_Nk[k])*(X[n,:]-mu[k,:]).T@(X[n,:]-mu[k,:])
            Sigma[:,:,k] = ssum
        if debug_mode == 1:
            print('Sigma')
            print(Sigma)

        # (3.4) re-estimate PComponents
        for k in range(num_clusters):
            PComponents[0,k] = exp(log_Nk[k]) / num_observations
        if (debug_mode == 1):
            print('PComponents')
            print(PComponents)

        # [3] Evaluate new log likelihood
        new_log_likelihood = 0
        for n in range(num_observations):
            log_cluster_lp = zeros((1, num_clusters))
            for k in range(num_clusters):
                log_cluster_lp[0,k] = log( mvn.pdf(X[n,:], mu[k,:], Sigma[:,:,k]) )
                log_cluster_lp[0,k] = log_cluster_lp[0,k] + log( PComponents[0, k] )

            # internal sum of log likelihood (adding probabibilites)
            log_sum_cluster_lp = log(npsum(exp(log_cluster_lp),1))
            # external sum of log likelihood (adding logarithms)
            new_log_likelihood = new_log_likelihood + log_sum_cluster_lp
        likelihooddiff = new_log_likelihood - old_log_likelihood

        if debug_mode == 1:
            print('new log likelihood is:')
            print(new_log_likelihood)
            print('likelihood difference')
            print(likelihooddiff)
            print('iteration')
            print(curr_iter)

        # [4] check the termination condition: multiplication by abs((newll)) is in
        # matlab code in toolbox/stats/stats/gmdistribution/private/gmcluster.m
        if (likelihooddiff >= 0 and likelihooddiff < 1e-6*abs(new_log_likelihood) ) or curr_iter > 100:
            print('algorithm ends after iterations')
            print(curr_iter)
            break
        else:
            curr_iter = curr_iter + 1
        if likelihooddiff < 0:
            print('Log likelihood is becoming smaller so we have to stop here - SOMETHING IS WRONG')
            print(likelihooddiff)
            break

        old_log_likelihood = new_log_likelihood
    # end of while loop

    ## --------- THE EM ALGORITHM ---------

    # (@) dysplay my model and matlab model
    if debug_mode == 1:
        print('my gmm:')
        print(mu)
        print(Sigma)
        print(PComponents)
    return mu, Sigma, PComponents

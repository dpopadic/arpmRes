from collections import namedtuple
import numpy as np
from datetime import datetime, timedelta

from CONFIG import IMGS_DIR
from matplotlib.pyplot import savefig
from numpy import meshgrid, pi, power as pow, isnan, diag
from numpy.linalg import svd, inv, det
from scipy.interpolate import LinearNDInterpolator, interp2d
from scipy.special import gamma
import sympy

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal


def multivariate_t_distribution(x,mu,Sigma,df):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
    '''

    d = Sigma.shape[0]
    Num = gamma(1. * (d+df)/2)
    Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(det(Sigma),1./2) *
              pow(1 + (1./df)*np.dot(x - mu,np.dot(inv(Sigma), (x - mu).T)), 1*(d+df)/2))
    d = 1*Num/Denom
    return diag(d)


def save_plot(ax, extension, scriptname, count=None):
    if ax.title._text != '':
        fname = ''.join([IMGS_DIR, scriptname, '_', ax.title._text.replace('.','').replace(' ','_'), '.', extension])
    else:
        fname = ''.join([IMGS_DIR, scriptname, '_%d.' % count, extension])
    savefig(fname, dpi=300)


def rref(x):
    res = sympy.Matrix(x).rref()
    return np.array(res[0].tolist(), dtype=float)


def nullspace(a, rtol=1e-5):
    u, s, v = svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()


def date_mtop(matlab_date):
    return datetime.fromordinal(int(matlab_date) - 366) + timedelta(days=float(matlab_date) % 1)\
           -timedelta(microseconds=divmod(timedelta(days=float(matlab_date) % 1).microseconds,1000)[1])


def time_mtop(matlab_time):
    matlab_time += 700000
    return datetime.fromordinal(int(matlab_time) - 366) + timedelta(days=float(matlab_time) % 1)\
           -timedelta(microseconds=divmod(timedelta(days=float(matlab_time) % 1).microseconds,1000)[1])


def datenum(s):
    d = datetime.strptime(s,'%d-%b-%Y')
    return d.toordinal()+366


def interpne(V,Xi,nodelist,method='linear'):
    if method =='linear':
        grids = meshgrid(*nodelist)
        flatten_grids = [grid.flatten() for grid in grids]
        cartcoord = list(zip(*flatten_grids))
        interp = LinearNDInterpolator(cartcoord, V.flatten('F'), fill_value=np.nan)
        Vpred = interp(*Xi)
        if isnan(Vpred):
            interp = interp2d(*flatten_grids, V, kind='cubic')
            Vpred = interp(*Xi)
    else:
        raise NotImplementedError('Method {method} is not available.....yet.'.format(method=method))
    return Vpred


def struct_to_dict(s, as_namedtuple=True):
    if as_namedtuple:
        if s.dtype.names:
            nt = namedtuple('db', s.dtype.names)
            nt = nt(**{x: s[x].flatten()[0] for x in s.dtype.names})
            return nt
    else:
        if s.dtype.names:
            return {x: s[x].flatten()[0] for x in s.dtype.names}


def nt_to_dict(s):
    d = {k: v for k, v in vars(s).items() if not k.startswith('_') and not isinstance(v, property)}
    return d


def convert_dtont(dictionary):
    for key, value in dictionary.items():
            if isinstance(value, dict):
                dictionary[key] = convert_dtont(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def matlab_percentile(in_data, percentiles):
    """
    Calculate percentiles in the way IDL and Matlab do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    Parameters
    ----------
    in_data: numpy.ndarray
        input data
    percentiles: numpy.ndarray
        percentiles at which to calculate the values

    Returns
    -------
    perc: numpy.ndarray
        values of the percentiles
    """

    data = np.sort(in_data)
    p_rank = 100.0 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc


def mkdt(d):
    ll = []
    for k, v in d.items():
        if isinstance(v,np.ndarray):
            ll.append((k,v.dtype))
        elif isinstance(v,(np.float, np.int, np.int64, str)):
            ll.append((k,type(v)))
        else:
            ll.append((k,mkdt(v)))
    return ll


def copy_values(d, A):
    if A.dtype.names:
        for n in A.dtype.names:
            copy_values(d[n], A[n])
    else:
        A[:] = d


def dtorec(d):
    ll = mkdt(d)
    out = np.zeros(len(d), dtype=ll)
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dtorec(v)
        else:
            out[k] = v
    return out
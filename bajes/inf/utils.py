#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import

import numpy as np
from ..pipe import cart2sph, sph2cart

import logging
logger = logging.getLogger(__name__)

# list/dict wrapper

def dict_2_list(dict,keys):
    return [dict[ki] for ki in keys]

def list_2_dict(x,keys):
    assert len(x) == len(keys)
    return {ki : xi for ki,xi in zip(keys,x)}

# autocorrelation

def autocorrelation(x):
    '''
        Compute autocorrelation funcion for a for an 1-dim array of samples,
        non partial method with numpy.fft

        Arguments:
        - x : The time series.
    '''

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    xp=x-np.mean(x)
    var=np.var(x)

    # do fft and ifft
    cf=np.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=np.fft.ifft(sf).real
    corr=corr/var/n

    return corr[:n]

def autocorr_function(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    Arguments:
        - x     : The time series.
        - axis  : The time axis of ``x``. Assumed to be the first axis if not specified, (optional)
        - fast  : If ``True``, only use the largest ``2^n`` entries for efficiency (default: False)
    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[tuple(m)]

def autocorr_integrated_time(x, axis=0, window=50, fast=True):
    """
    Estimate the integrated autocorrelation time of a time series.
    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times.

    Arguments:
        - x         : The time series.
        - axis      : The time axis of ``x``. Assumed to be the first axis if not specified, (optional)
        - window    : The size of the window to use. (optional, default 50)
        - fast      : If ``True``, only use the largest ``2^n`` entries for efficiency (default: False)
    """
    # Compute the autocorrelation function.
    f = autocorr_function(x, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2*(f[1:window]).sum()

    # N-dimensional case.
    m       = [slice(None), ] * len(f.shape)
    m[axis] = slice(1, window)
    tau     = 1 + 2*(f[tuple(m)]).sum(axis=axis)

    return tau

# thermodinamic evidence integration (ptmcmc)

def thermodynamic_integration_log_evidence(betas, logls):
    """
    Thermodynamic integration estimate of the evidence.

    Arguments:
        - betas : list, the inverse temperatures to use for the quadrature.
        - logls : list, the mean log-likelihoods corresponding to ``betas`` to use for computing the thermodynamic evidence.

    Return:
        - logZ  : float, estimated log-evidence
        - dlogZ : float, error associated with the finite number of temperatures
    """
    if len(betas) != len(logls):
        raise ValueError('Need the same number of log(L) values as temperatures.')

    order = np.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]

    betas0 = np.copy(betas)
    if betas[-1] != 0:
        betas = np.concatenate((betas0, [0]))
        betas2 = np.concatenate((betas0[::2], [0]))

        # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        logls = np.concatenate((logls, [logls[-1]]))
    else:
        betas2 = np.concatenate((betas0[:-1:2], [0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

    logZ = -np.trapz(logls, betas)
    logZ2 = -np.trapz(logls2, betas2)
    return logZ, np.abs(logZ - logZ2)

# mcmc step estimation (dynesty)

def estimate_nmcmc(accept_ratio, old_act, maxmcmc, safety=5, tau=None):
    """
        Estimate autocorrelation length of chain using acceptance fraction,
        extracted from bilby, https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/sampler/dynesty.py

        Arguments:
        - accept_ratio  : float [0, 1], ratio of the number of accepted points to the total number of points
        - old_act       : int, the ACT of the last iteration
        - maxmcmc       : int, the maximum length of the MCMC chain
        - safety        : int, safety factor
        - tau           : int, ACT if given, otherwise estimated (optional)

    """
    if tau is None:
        tau = maxmcmc / safety

    if accept_ratio == 0.0:
        Nmcmc_exact = (1 + 1 / tau) * old_act
    else:
        Nmcmc_exact = ((1. - 1. / tau) * old_act + (safety / tau) * (2. / accept_ratio - 1.))
        Nmcmc_exact = float(min(Nmcmc_exact, maxmcmc))
    return max(safety, int(Nmcmc_exact))

# proposal helpers

def list_in_bounds(x, bounds):
    return all(list(map(lambda xi,bi : bi[0]<=xi<=bi[1], x,bounds)))

def apply_bounds(q, periodic_or_reflective, bounds):

    # if params in bounds, return params
    if list_in_bounds(q, bounds):
        return q
    # else, move params in bounds
    else:
        bound_funcs = [move_in_bound_periodic if ri else move_in_bound_reflective for ri in periodic_or_reflective]
        return np.array([bound_funcs[i](q[i], bounds[i][0], bounds[i][1]) for i in range(len(q))])

def move_in_bound_reflective(x, low, up):
    """
        Move value in bounds assuming reflective boundaries
    """
    if x < low :
        return 2.*low-x
    elif x > up :
        return 2.*up-x
    else:
        return x

def move_in_bound_periodic(x, low, up):
    """
        Move value in bounds assuming periodic boundaries
    """
    if x < low:
        return up - np.abs(x - low)%(up - low)
    elif x > up:
        return (x - low)%(up - low) + low
    else:
        return x

def reflect_skyloc_3dets(ra,dec,refvec,refloc):
    """
        Project skyloc orthogonally to the plane of the first 3 dets,
        as proposed in J. Veitch et al.
        arXiv:0911.3820v2 [astro-ph.CO] (2010)
        arXiv:1409.7215v2 [gr-qc] (2015)
    """
    locnorm = np.sqrt(refloc[0]**2 + refloc[1]**2 + refloc[2]**2)
    refloc /= locnorm

    x0          = np.array(sph2cart(1,np.pi/2-dec,ra))
    x1          = x0 - 2*refvec*(np.dot(refvec,x0-refloc))
    r, th, ph   = cart2sph(x1[0],x1[1],x1[2])
    dtshift     = np.real(locnorm * np.dot(refloc,x1-x0)) / 299792458.0

    # take into account Earth rotation,
    # move time_shift in radiants and substract to right ascenscion
    # 86400 s = 24h = 2 pi
    ph  -= dtshift * 2*np.pi/86400.

    return np.real(ph) , np.pi/2.-th, dtshift

def reflect_skyloc_2dets(ra,dec,refvec,refloc):
    """
        Compute rotation of skyposition of a random angle
        around center of the detectors,
        as proposed in J. Veitch et al.
        arXiv:0911.3820v2 [astro-ph.CO] (2010)
        arXiv:1409.7215v2 [gr-qc] (2015)
    """
    from ..pipe import rotation_matrix
    # x_,y_,z_    = sph2cart(1,np.pi/2-dec,ra)
    # x0          = np.array([x_,y_,z_])
    x0          = np.array(sph2cart(1,np.pi/2-dec,ra))
    m           = rotation_matrix(refvec, np.random.normal(0,2*np.pi))
    x1          = np.dot(m,x0)
    r, th, ph   = cart2sph(x1[0],x1[1],x1[2])
    dtshift     = np.dot(refloc,x1-x0) / 299792458.0

    # take into account Earth rotation,
    # move time_shift in radiants and substract to right ascenscion
    # 86400 s = 24h = 2 pi
    ph  -= dtshift * 2*np.pi / 86400

    return ph , np.pi/2 - th, dtshift

def project_all_extrinsic(dets, ra, dec, iota, dist, psi, tshift, ra_new, dec_new, tshift_new, t_gps):
    """
        Compute projection of extrinsic parameters
        given a new skylocation,
        as proposed in V. Raymond and W.M. Farr
        arXiv:1402.0053v1 [gr-qc] (2014)
        This formulation ssumines that:
        F+(psi1) = F+(psi0) * cos(2 * psi1) - Fx(psi0) * sin(2 * psi1)
        Fx(psi1) = Fx(psi0) * cos(2 * psi1) + F+(psi0) * sin(2 * psi1)
    """

    ifos    = list(dets.keys())
    R2      = {}
    x2      = {}
    y2      = {}

    for ifo in ifos:
        cosi        = np.cos(iota)
        fp, fc      = dets[ifo].antenna_pattern(ra, dec, psi, t_gps+tshift)
        x, y        = dets[ifo].antenna_pattern(ra_new,
                                                dec_new,
                                                psi,
                                                t_gps+tshift_new)

        Aplus       = 0.5*(1+cosi*cosi)*fp/dist
        Across      = cosi*fc/dist
        R2[ifo]     = Aplus*Aplus + Across*Across
        x2[ifo]     = x*x
        y2[ifo]     = y*y

    i1 = ifos[0]
    i2 = ifos[1]
    i3 = ifos[2]

    A = R2[i3]*x2[i2]*y2[i1] - R2[i2]*x2[i3]*y2[i1] - R2[i3]*x2[i1]*y2[i2] + R2[i1]*x2[i3]*y2[i2] + R2[i2]*x2[i1]*y2[i3] - R2[i1]*x2[i2]*y2[i3]
    B = - R2[i3]*x2[i2]*np.sqrt(x2[i1]*y2[i1]) + R2[i2]*x2[i3]*np.sqrt(x2[i1]*y2[i1]) + R2[i3]*x2[i1]*np.sqrt(x2[i2]*y2[i2]) - R2[i1]*x2[i3]*np.sqrt(x2[i2]*y2[i2]) + R2[i3]*y2[i1]*np.sqrt(x2[i2]*y2[i2]) - R2[i3]*y2[i2]*np.sqrt(x2[i1]*y2[i1]) - R2[i2]*x2[i1]*np.sqrt(x2[i3]*y2[i3]) + R2[i1]*x2[i2]*np.sqrt(x2[i3]*y2[i3]) - R2[i2]*y2[i1]*np.sqrt(x2[i3]*y2[i3]) + R2[i1]*y2[i2]*np.sqrt(x2[i3]*y2[i3]) + R2[i2]*y2[i3]*np.sqrt(x2[i1]*y2[i1]) - R2[i1]*y2[i3]*np.sqrt(x2[i2]*y2[i2])
    B_A = B/A

    psi_new1 = 0.5*np.arctan2(B - A*np.sqrt(1+B_A*B_A), A)
    psi_new2 = 0.5*np.arctan2(B + A*np.sqrt(1+B_A*B_A), A)

    fp2_new1  = {}
    fc2_new1  = {}
    fp2_new2  = {}
    fc2_new2  = {}

    for ifo in ifos:
        fp1, fc1      = dets[ifo].antenna_pattern(ra_new,
                                                  dec_new,
                                                  psi_new1,
                                                  t_gps+tshift_new)
        fp2, fc2      = dets[ifo].antenna_pattern(ra_new,
                                                  dec_new,
                                                  psi_new2,
                                                  t_gps+tshift_new)

        fp2_new1[ifo] = fp1*fp1
        fc2_new1[ifo] = fc1*fc1
        fp2_new2[ifo] = fp2*fp2
        fc2_new2[ifo] = fc2*fc2

    psi_new = psi_new1
    fp2_new = fp2_new1
    fc2_new = fc2_new1

    cos2i_new = (R2[i1]*(2*fc2_new[i2]+fp2_new[i2]) - R2[i2]*(2*fc2_new[i1]+fp2_new[i1]))/(fp2_new[i1]*R2[i2]-fp2_new[i2]*R2[i1]) - 2 * np.sqrt((fc2_new[i2]*R2[i1] - fc2_new[i1]*R2[i2])*(R2[i1]*(fc2_new[i2]+fp2_new[i2]) - R2[i2]*(fc2_new[i1]+fp2_new[i1])))/(fp2_new[i2]*R2[i1] - fp2_new[i1]*R2[i2])**2.

    if cos2i_new < 0. or cos2i_new > 1. or np.isnan(cos2i_new):

        psi_new = psi_new2
        fp2_new = fp2_new2
        fc2_new = fc2_new2

        cos2i_new = (R2[i1]*(2*fc2_new[i2]+fp2_new[i2]) - R2[i2]*(2*fc2_new[i1]+fp2_new[i1]))/(fp2_new[i1]*R2[i2]-fp2_new[i2]*R2[i1]) - 2 * np.sqrt((fc2_new[i2]*R2[i1] - fc2_new[i1]*R2[i2])*(R2[i1]*(fc2_new[i2]+fp2_new[i2]) - R2[i2]*(fc2_new[i1]+fp2_new[i1])))/(fp2_new[i2]*R2[i1] - fp2_new[i1]*R2[i2])**2.

    cosi_new = np.sqrt(cos2i_new)
    iota_new = np.arccos(cosi_new)


    dist_new = np.sqrt((((0.5*(1.+cos2i_new))**2.)*fp2_new[i1] + cos2i_new*fc2_new[i1])/R2[i1])

    return dist_new, iota_new, psi_new

# initialize parameters

def get_parameter_distribution_from_string(name, type, min, max, kwargs):

    if type == 'uniform':
        from . import UniformProbability
        return UniformProbability(min=min, max=max)

    elif type == 'linear':
        from . import LinearProbability
        return LinearProbability(min=min, max=max)

    elif type == 'quadratic':
        from . import QuadraticProbability
        return QuadraticProbability(min=min, max=max)

    elif type == 'power-law':
        if 'deg' not in list(kwargs.keys()):
            raise AttributeError("Please include 'deg' in key-word arguments for power-law probability density ({} parameter).".format(name))
        if kwargs['deg'] == -1:
            raise AttributeError("Unable to set power-law distribution with degree -1 for {} parameter. Please use log-uniform distribution.".format(name))
        if min < 0:
            raise AttributeError("Unable to set power-law distribution for {} parameter, power-law prior does not support negative bounds.".format(name))
        from . import PowerLawProbability
        return PowerLawProbability(min=min, max=max, deg=kwargs['deg'])

    elif type == 'triangular':
        if 'mode' not in list(kwargs.keys()):
            raise AttributeError("Please include 'mode' in key-word arguments for triangular probability density ({} parameter).".format(name))
        if kwargs['mode'] > max or kwargs['mode'] < min :
            raise AttributeError("Unable to set triangular distribution for {} parameter, mode for triangular prior is outside prior bounds.".format(name))
        from . import TriangularProbability
        return TriangularProbability(min=min, max=max, mode=kwargs['mode'])

    elif type == 'log-uniform':
        if min <= 0:
            raise AttributeError("Unable to set log-uniform distribution for {} parameter, bound is including the origin (non-integrable point).".format(name))
        from . import LogUniformProbability
        return LogUniformProbability(min=min, max=max)

    elif type == 'cosinusoidal':
        if min < -np.pi/2 or max > np.pi/2:
            logger.warning("Cosinusoidal distribution is tuned to work within the range [-pi/2, +pi/2]")
        from . import CosinusoidalProbability
        return CosinusoidalProbability(min=min, max=max)

    elif type == 'sinusoidal':
        if min < 0 or max > np.pi:
            logger.warning("Cosinusoidal distribution is tuned to work within the range [0, pi]")
        from . import SinusoidalProbability
        return SinusoidalProbability(min=min, max=max)

    elif type == 'exponential':
        if 'tau' not in list(kwargs.keys()):
            raise AttributeError("Please include 'tau' in key-word arguments for exponential probability density ({} parameter).".format(name))
        if kwargs['tau'] <= 0:
            raise AttributeError("Unable to set exponential distribution for {} parameter, decay time `tau` must be a real posivite value.".format(name))
        from . import ExponentialProbability
        return ExponentialProbability(min=min, max=max, tau=kwargs['tau'])

    elif type == 'normal':
        kwarg = kwargs
        if 'mu' not in list(kwargs.keys()) or 'sigma' not in list(kwargs.keys()):
            raise AttributeError("Please include 'mu' and 'sigma' in key-word arguments for normal probability density ({} parameter).".format(name))
        if kwargs['sigma'] <= 0 :
            raise AttributeError("Unable to set normal distribution for {} parameter, `sigma` must be a real positive value.".format(name))
        from . import NormalProbability
        return NormalProbability(min=min, max=max, mu=kwargs['mu'], sigma=kwargs['sigma'])

    else:
        from . import __known_probs__
        raise AttributeError("Unable to set probability distribution for {} parameter. Unknown distribution {}, please use one of the followings: {}".format(name, type, __known_probs__))

def wrap_exp_log_prior(x, f, kw):
    return np.exp(f(x, **kw))

def initialize_param_from_func(name, min, max, func, kwarg={}, ngrid=2000, kind='linear'):
    """
        Compute probability density and cumulative probability function
        for the given parameter distribution using quadrature approximation
        (scipy.integrate.quad) and interpolation method (scipy.interpolate.interp1d)
    """

    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    # define grid
    ax  = np.linspace(min,max,ngrid)

    # estimate log-PDF and CDF
    pdf = [func(pj, **kwarg) for pj in ax]
    cdf = [quad(wrap_exp_log_prior, min, pj, args=(func,kwarg))[0] for pj in ax]

    # check monotonicity
    if np.sum(np.diff(cdf)<0):
        logger.warning("Unable to estimate customized prior for {} parameter with quadrature rule, switching to trapezoidal approximation".format(name))

        # recompute CDF using trapezoidal rule, cruder approximation but improves discontinuity
        dx      = ax[1] - ax[0]
        exp_pdf = np.exp(pdf)
        cdf     = [np.trapz(exp_pdf[:(i+1)], dx = dx) for i in range(ngrid)]

        # recheck monotonicity
        if np.sum(np.diff(cdf)<0):
            logger.error("Unable to estimate customized prior distribution for {} parameter, cumulative prior is not monotonic.".format(name))
            raise RuntimeError("Unable to estimate customized prior distribution for {} parameter, cumulative prior is not monotonic.".format(name))

    # check NaNs
    if np.sum(np.isnan(cdf))+np.sum(np.isnan(pdf)):
        logger.error("Unable to estimate customized prior distribution for {} parameter, NaN occured.".format(name))
        raise RuntimeError("Unable to estimate customized prior distribution for {} parameter, NaN occured.".format(name))

    # check Infs
    if np.sum(np.isinf(cdf)):
        logger.error("Unable to estimate customized prior distribution for {} parameter, Inf occured.".format(name))
        raise RuntimeError("Unable to estimate customized prior distribution for {} parameter, Inf occured.".format(name))

    # rescale to unit
    integral    = cdf[-1]
    pdf         = np.array(pdf)-np.log(integral)
    cdf         = np.array(cdf)/integral

    # interpolators
    log_pdf_intrp   = interp1d(x=ax,  y=pdf,  bounds_error=False, kind=kind, fill_value=(-np.inf,-np.inf))
    qnt_intrp       = interp1d(x=cdf, y=ax,   bounds_error=True,  kind=kind)
    cdf_intrp       = interp1d(x=ax,  y=cdf,  bounds_error=False, kind=kind, fill_value=(0,1))

    from . import CustomProbability
    return CustomProbability(log_density=log_pdf_intrp, cumulative=cdf_intrp, quantile=qnt_intrp)


# prior sampler

def prior_sampler(prior, size):
    return np.array([prior.prior_transform(np.random.uniform(0.,1.,prior.ndim)) for _ in range(size)])

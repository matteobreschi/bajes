#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import numpy as np

from scipy.special import erf, erfinv
from collections import namedtuple

from .prior import Prior, Parameter, JointPrior
from .likelihood import Likelihood, Posterior, JointLikelihood

__known_samplers__     = ['emcee', 'ptmcmc', 'cpnest', 'ultranest', 'dynesty', 'dynesty-dyn']

def Sampler(engine, model, **kwargs):
    """
        Initialize a sampler object

        Arguments:
        engine  : str, specify the kind of sampler
                  mcmc, ptmcmc, nest, cpnest, dynest
        model   : list or bajes.inf.likelihood.Posterior,
                  if list, should contain 2 elements,
                  [bajes.inf.prior.Prior, bajes.inf.likelihood.Likelihood]
        kwargs  : further arguments to be passed to the sampler
    """

    if isinstance(model, list):

        if isinstance(model[0], Prior):
            pr = model[0]
            lk = model[1]
            posterior = Posterior(like=lk, prior=pr)

        elif isinstance(model[1], Prior):
            pr = model[1]
            lk = model[0]
            posterior = Posterior(like=lk, prior=pr)

        else:
            raise ValueError("Unable to define {} sampler. The model-list must contain two arguments, e.g. [bajes.inf.prior.Prior, bajes.inf.likelihood.Likelihood]".format(engine))

    elif isinstance(model, Posterior):
        posterior = model
    else:
        raise ValueError("Unable to define {} sampler. The model should be a bajes.inf.likelihood.Posterior object or a list containing [bajes.inf.prior.Prior, bajes.inf.likelihood.Likelihood]".format(engine))

    if engine == 'emcee':
        from .sampler.emcee import SamplerMCMC as sampler
    elif engine == 'ptmcmc':
        from .sampler.ptmcmc import SamplerPTMCMC as sampler
    elif engine == 'cpnest':
        from .sampler.cpnest import _WrapSamplerCPNest as sampler
    elif engine == 'dynesty':
        from .sampler.dynesty import SamplerDynesty as sampler
    elif engine == 'dynesty-dyn':
        from .sampler.dynesty import SamplerDynestyDynamic as sampler
    elif engine == 'ultranest':
        from .sampler.ultranest import SamplerUltraNest as sampler
    else:
        raise ValueError("Unable to define {} sampler. Please use one of the following: {}".format(engine, ', '.join(__known_samplers__)))

    return sampler(engine, posterior, **kwargs)

# namedtuple for customized probability
CustomProbability    = namedtuple("CustomProbability",    ("log_density","cumulative","quantile"),    defaults=(None,None,None) )

# Known probability functions
__known_probs__     = ['uniform', 'linear', 'quadratic', 'power-law',
                       'triangular', 'cosinusoidal', 'sinusoidal',
                       'log-uniform', 'exponential', 'normal']

class UniformProbability:
    """
        Uniform probability methods
        ______
        x   : float or array
        min : lower bound
        max : upper bound
    """

    def __init__(self, min, max):
        self._norm = max-min
        self._lognorm = np.log(self._norm)
        self._min = min

    def log_density(self, x):
        return -self._lognorm

    def cumulative(self, x):
        return (x-self._min)/self._norm

    def quantile(self, x):
        return x*self._norm + self._min

class LinearProbability:
    """
        Linear probability methods
        ______
        x   : float or array
        min : lower bound
        max : upper bound
    """

    def __init__(self, min, max):
        self._norm = 0.5*(max**2. - min**2.)
        self._lognorm = np.log(self._norm)
        self._min2 = min**2.

    def log_density(self, x):
        return np.log(x) - self._lognorm

    def cumulative(self, x):
        return 0.5*(x**2. - self._min2)/self._norm

    def quantile(self, x):
        return np.sqrt(x*2.*self._norm + self._min2)

class QuadraticProbability:
    """
        Quadratic probability methods
        ______
        x   : float or array
        min : lower bound
        max : upper bound
    """

    def __init__(self, min, max):
        self._norm = (max**3. - min**3.)/3.
        self._lognorm = np.log(self._norm)
        self._min3 = min**3.

    def log_density(self, x):
        return 2.*np.log(x) - self._lognorm

    def cumulative(self, x):
        return (x**3. - self._min3)/(self._norm*3.)

    def quantile(self, x):
        return (x*3.*self._norm + self._min3)**(1./3.)

class PowerLawProbability:
    """
        Quadratic probability methods
        ______
        x   : float or array
        min : lower bound
        max : upper bound
        """

    def __init__(self, min, max, deg):

        self._deg   = deg
        self._degp1 = deg + 1.

        max2degp1 = max**self._degp1
        self._min2degp1 = min**self._degp1

        self._norm = (max2degp1 - self._min2degp1)/self._degp1
        self._lognorm = np.log(self._norm)

    def log_density(self, x):
        return self._deg*np.log(x) - self._lognorm

    def cumulative(self, x):
        x2deg = x**self._degp1
        return (x2deg - self._min2degp1)/(self._norm*self._degp1)

    def quantile(self, x):
        return (x*self._norm*self._degp1+ self._min2degp1)**(1./self._degp1)

class TriangularProbability:
    """
        Triangular probability methods
        ______
        x    : float or array
        min  : lower bound
        max  : upper bound
        mode : modal value
    """

    def __init__(self, min, max, mode):

        self._min   = min
        self._max   = max
        self._mode  = mode

    def log_density(self, x):
        if x<= self._mode:
            return np.log(2.*(x-self._min)/(self._max-self._min)/(self._mode-self._min))
        else:
            return np.log(2.*(self._max-x)/(self._max-self._min)/(self._max-self._mode))

    def cumulative(self, x):
        if x<= self._mode:
            return (x-self._min)**2./(self._max-self._min)/(self._mode-self._min)
        else:
            return 1. - (self._max-x)**2./(self._max-self._min)/(self._max-self._mode)

    def quantile(self, x):
        if x <= (self._mode-self._min)/(self._max-self._min):
            return self._min + np.sqrt(x*(self._max-self._min)*(self._mode-self._min))
        else:
            return self._max - np.sqrt((x-1.)*(self._max-self._min)*(self._max-self._mode))

class LogUniformProbability:
    """
        Log-uniforn probability methods
        ______
        x    : float or array
        min  : lower bound
        max  : upper bound
    """

    def __init__(self, min, max):

        self._min       = min
        self._r         = max/min
        self._norm      = np.log(self._r)
        self._lognorm   = np.log(self._norm)

    def log_density(self, x):
        return -np.log(x) - self._lognorm

    def cumulative(self, x):
        return np.log(x/self._min)/self._norm

    def quantile(self, x):
        return self._min*(self._r**x)

class SinusoidalProbability:
    """
        Sinusoidal probability methods
        ______
        x    : float or array
        min  : lower bound
        max  : upper bound
        """

    def __init__(self, min, max):

        self._cosmin    = np.cos(min)
        self._norm      = np.cos(min) - np.cos(max)
        self._lognorm   = np.log(self._norm)

    def log_density(self, x):
        return np.log(np.abs(np.sin(x))) - self._lognorm

    def cumulative(self, x):
        return (self._cosmin - np.cos(x))/self._norm

    def quantile(self, x):
        return np.arccos(self._cosmin - self._norm*x)

### semicircle
##def log_density_semicircle(x, min, max):
##    return 0.5*np.log(max*max-x*x)-np.log(np.pi*max*max/2.)
##
##def cumulative_semicircle(x, min, max):
##
##def quantile_semicircle(x, min, max):

class CosinusoidalProbability:
    """
        Sinusoidal probability methods
        ______
        x    : float or array
        min  : lower bound
        max  : upper bound
        """

    def __init__(self, min, max):

        self._sinmin    = np.sin(min)
        self._norm      = np.sin(max) - np.sin(min)
        self._lognorm   = np.log(self._norm)

    def log_density(self, x):
        return np.log(np.abs(np.cos(x))) - self._lognorm

    def cumulative(self, x):
        return (np.sin(x) - self._sinmin)/self._norm

    def quantile(self, x):
        return np.arcsin(self._norm*x + self._sinmin)

class ExponentialProbability:
    """
        Exponential probability methods
        ______
        x    : float or array
        min  : lower bound
        max  : upper bound
        tau  : decay parameter
        """

    def __init__(self, min, max, tau):

        self._tau       = tau
        self._norm      = (np.exp(-min/tau) - np.exp(-max/tau))*tau
        self._lognorm   = np.log(self._norm)
        self._expmin    = np.exp(-min/tau)

    def log_density(self, x):
        return - x/self._tau - self._lognorm

    def cumulative(self, x):
        n = self._norm/self._tau
        return (self._expmin - np.exp(-x/tau))/n

    def quantile(self, x):
        n = self._norm/self._tau
        return -tau*np.log(self._expmin - n*x)

class NormalProbability:
    """
        Truncated gaussian probability
        ______
        x       : float or array
        min     : lower bound
        max     : upper bound
        mu      : mean
        sigma   : stdev
    """
    def __init__(self, min, max, mu, sigma):

        self._mu        = mu
        self._sigma     = sigma
        self._sqrt2     = np.sqrt(2.)

        a   = (min - mu)/sigma
        b   = (max - mu)/sigma

        self._zeta      = 0.5*(erf(b/self._sqrt2) - erf(a/self._sqrt2))
        self._norm      = np.sqrt(2*np.pi)*sigma*self._zeta
        self._lognorm   = np.log(self._norm)
        self._pmin      = 0.5*(1. + erf(a/self._sqrt2))

    def log_density(self, x):
        return -0.5*((x-self._mu)/self._sigma)**2. - self._lognorm

    def cumulative(self, x):
        xi = (x-mu)/sigma
        px = 0.5*(1.+erf(xi/self._sqrt2))
        return (px-self._pmin)/self._zeta

    def quantile(self, x):
        _x = 2.*(self._zeta*x + self._pmin)-1.
        return self._mu + self._sigma*self._sqrt2*erfinv(_x)

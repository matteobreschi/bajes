from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from .utils import list_2_dict

class Likelihood(object):
    """
        Wrapper object for generic likelihood function
    """

    def __init__(self, func=None, args=[], kwargs={}):
        
        self.logZ_noise = 0.
        self._func      = func
        self._args      = args
        self._kwargs    = kwargs
    
    def log_like(self, x):
        
        if self._func == None:
            logger.error("Log-likelihood method has been called without specifying the likelihood form. Please implement your Likelihood.")
            raise RuntimeError("Log-likelihood method has been called without specifying the likelihood form. Please implement your Likelihood.")
        else:
            logl = self._func(x, *self._args, **self._kwargs)

        if np.isnan(logl):
            logger.error("Likelihood method returned NaN for the set of parameters: {}.".format(x))
            raise RuntimeError("Likelihood method returned NaN for the set of parameters: {}.".format(x))

        return logl

class Posterior(object):
    """
        Posterior object
    """

    def __init__(self, like, prior):
        """
            Initialize Posterior object
            
            Arguments:
            - like  : a bajes.inf.Likelihood object
            - prior : a bajes.inf.Prior object
        """
        self.like = like
        self.prior = prior
    
    def prior_transform(self, u):
        return self.prior.prior_transform(u)

    def log_like(self, x):
        p   = self.prior.this_sample(list_2_dict(x, self.prior.names))
        return self.like.log_like(p)

    def log_prior(self, x):
        if self.prior.in_bounds(x):
            return self.prior.log_prior(x)
        else:
            return -np.inf

    def log_post(self, x):
        if self.prior.in_bounds(x):
            p   = self.prior.this_sample(list_2_dict(x, self.prior.names))
            lnp = self.prior.log_prior(x)
            lnl = self.like.log_like(p)
            return lnp + lnl
        else:
            return -np.inf

    def log_likeprior(self, x):
        if self.prior.in_bounds(x):
            p   = self.prior.this_sample(list_2_dict(x, self.prior.names))
            lnp = self.prior.log_prior(x)
            lnl = self.like.log_like(p)
            return lnl, lnp
        else:
            return 0, -np.inf

class JointLikelihood(Likelihood):
    """
        Joint likelihood object
    """

    def __init__(self, likes):
        """
            Initialize Posterior object
            
            Arguments:
            - likes : list of bajes.inf.Likelihood objects
        """

        # run standard initialization
        super(JointLikelihood, self).__init__()

        # store likelihoods
        self.likes = likes

    def log_like(self, x):
        return sum([l.log_like(x) for l in self.likes])

from __future__ import division, unicode_literals, absolute_import
import numpy as np
import signal
import tracemalloc
import os

import logging
logger = logging.getLogger(__name__)

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from itertools import repeat

from . import SamplerBody
from ..utils import apply_bounds, autocorr_integrated_time, thermodynamic_integration_log_evidence
from ...pipe import eval_func_tuple, ensure_dir
from .proposal import ModelTuple, _init_proposal_methods

def initialize_proposals(like, priors, use_slice=False, use_gw=False):

    props = {'eig': 25., 'dif': 25., 'str': 20., 'wlk': 15., 'kde': 10., 'pri': 5.}
    prop_kwargs  = {}

    if use_slice:
        props['slc'] = 30.

    if use_gw:
        props['gwt'] = 15.
        prop_kwargs['like'] = like
        prop_kwargs['dets'] = like.dets

    return BajesPTMCMCProposal(priors, props, **prop_kwargs)

def default_beta_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
    arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:

    Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
    this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
    <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
    ``ntemps`` is also specified.

    :param ndim:
        The number of dimensions in the parameter space.

    :param ntemps: (optional)
        If set, the number of temperatures to generate.

    :param Tmax: (optional)
        If set, the maximum temperature for the ladder.

    Temperatures are chosen according to the following algorithm:

    * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
      information).
    * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
      posterior would have a 25% temperature swap acceptance ratio.
    * If ``Tmax`` is specified but not ``ntemps``:

      * If ``Tmax = inf``, raise an exception (insufficient information).
      * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.

    * If ``Tmax`` and ``ntemps`` are specified:

      * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
      * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.

    """

    if type(ndim) != int or ndim < 1:
        raise ValueError('Invalid number of dimensions specified.')
    if ntemps is None and Tmax is None:
        raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
    if Tmax is not None and Tmax <= 1:
        raise ValueError('``Tmax`` must be greater than 1.')
    if ntemps is not None and (type(ntemps) != int or ntemps < 1):
        raise ValueError('Invalid number of temperatures specified.')

    tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                      2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                      2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                      1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                      1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                      1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                      1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                      1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                      1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                      1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                      1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                      1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                      1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                      1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                      1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                      1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                      1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                      1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                      1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                      1.26579, 1.26424, 1.26271, 1.26121,
                      1.25973])

    if ndim > tstep.shape[0]:
        # An approximation to the temperature step at large
        # dimension
        tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
    else:
        tstep = tstep[ndim-1]

    appendInf = False
    if Tmax == np.inf:
        appendInf = True
        Tmax = None
        ntemps = ntemps - 1

    if ntemps is not None:
        if Tmax is None:
            # Determine Tmax from ntemps.
            Tmax = tstep ** (ntemps - 1)
    else:
        if Tmax is None:
            raise ValueError('Must specify at least one of ``ntemps'' and finite ``Tmax``.')

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with infinity.
        betas = np.concatenate((betas, [0]))

    return betas

class BajesPTMCMCProposal(object):

    def __init__(self, priors, props=None, **kwargs):
        """
            Proposal cycle object for MCMC sampler
            Arguments:
            - prior   : prior object
            - props   : dictionary specifying the proposal settings,
                        the dictionary keys determine which proposal are included,
                        while the value determines the relative weight.
                        obs. the weights are rescaled such that their sum is equal to 1.
                        The available proposals are:
                         - 'eig' = eigen-vector proposal
                         - 'dif' = differential evolution proposal
                         - 'str' = stretch proposal
                         - 'kde' = gaussian-kde proposal
                         - 'wlk' = random-walk proposal
                         - 'pri' = prior proposal
                         - 'slc' = slice proposal
                         - 'gwt' = gw-targeted proposal
                         (default {'eig': 25., 'dif': 25., 'str': 20, 'wlk': 15, 'kde': 10, 'pri': 5})
            - kwargs   : additional arguments,
                         - like      : likelihood object, if gwt
                         - dets      : dictionary of detectors, if gwt
                         - subset    : number of sampler in the subset, if wlk (default 25)
                         - stretch   : stretching factor, if str (default 2)
                         - gamma     : scale parameter, if dif (default 2.38/sqrt(ndim))
                         - bw_method : band-width method, if kde (default None)
                         - rnd       : random stdev, if gwt (default 1e-5)
                         - mu        : initial scale parameter, if slc (default 1)
                         - threshold : last adaptive iteration, if slc (default 1500)
                         - cov       : initial covariance, if eig (default identity)
         """

        self.names                  = priors.names
        self.bounds                 = priors.bounds
        self.ndim                   = len(self.names)
        self.period_reflect_list    = priors.periodics

        self._proposals, self._weights = _init_proposal_methods(priors, props=props, **kwargs)

    def propose(self, s, c, p, model):
        Nt  = len(s)
        _p  = model.random.choice(self._proposals, size=Nt, p=self._weights)
        q_f = [fn.get_proposal(si, [ci], pi, model) for fn, si, ci, pi in zip(_p,s,c,p)]
        q_f = list(zip(*q_f))
        return q_f[0], q_f[1]

class _PTMCMC(object):

    def __init__(self, nwalkers, posterior, proposal,
                 ntemps=None, Tmax=None, betas=None,
                 adaptation_lag=10000, adaptation_time=100,
                 random=None, pool=None):

        if random is None:
            self._random = np.random
        else:
            self._random = random

        self.nwalkers        = nwalkers
        self.dim             = len(posterior.prior.names)
        self.adaptation_time = adaptation_time
        self.adaptation_lag  = adaptation_lag

        # set boundaries
        self._period_reflect_list   = posterior.prior.periodics
        self._bounds                = posterior.prior.bounds

        # Set temperature ladder.  Append beta=0 to generated ladder.
        if betas is not None:
            self._betas = np.array(betas).copy()
        else:
            self._betas = default_beta_ladder(self.dim, ntemps=ntemps, Tmax=Tmax)

        # Make sure ladder is ascending in temperature.
        self._betas[::-1].sort()

        if self.nwalkers % 2 != 0:
            raise ValueError('The number of walkers must be even.')

        # get pool and map
        self.pool = pool

        if self.pool == None:
            self.mapf = map
        else:
            self.mapf = self.pool.map

        # avoid default function wrapper
        self._likeprior = posterior.log_likeprior

        # set proposals
        self._proposal = proposal

        # initialize sampler arguments
        self.__start__()

    def __start__(self):
        """
        Instantiate the ``time``, ``chain``, ``logposterior``,
        ``loglikelihood``,  ``acceptance_fraction``,
        ``tswap_acceptance_fraction`` stored properties.
        """

        # Reset chain.
        self._chain = None
        self._logposterior = None
        self._loglikelihood = None

        # Reset history.
        self._acceptance_history        = []
        self._tswap_acceptance_history  = []
        self._chain_history             = []
        self._loglikelihood_history     = []
        self._logposterior_history      = []

        # Reset sampler state.
        self._time = 0
        self._p0 = None
        self._logposterior0 = None
        self._loglikelihood0 = None

        self.nswap = np.zeros(self.ntemps, dtype=np.float)
        self.nswap_accepted = np.zeros(self.ntemps, dtype=np.float)

        self.nprop = np.zeros((self.ntemps, self.nwalkers), dtype=np.float)
        self.nprop_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=np.float)

    def __getstate__(self):
        state               = self.__dict__.copy()
        state['_random']    = None
        state['pool']       = None
        state['mapf']       = None
        return state

    def _evaluate(self, ps):
        logger.debug("Evaluating probabilities...")
        results = list(self.mapf(self._likeprior, ps.reshape((-1, self.dim))))

        # logL, logpr
        logl    = np.fromiter((r[0] for r in results), np.float, count=len(results)).reshape((self.ntemps, -1))
        logp    = np.fromiter((r[1] for r in results), np.float, count=len(results)).reshape((self.ntemps, -1))
        logger.debug("Probabilities evaluated.")

        return logl, logp

    def _like_func(self, s):
        l_p = [self._likeprior(s[i]) for i in range(len(s))]
        l_p = list(zip(*l_p))
        return np.array(l_p[0]), None

    def _propose(self, p, logpost, logl):

        # wrap useful methods
        model = ModelTuple(map_fn=self.mapf, compute_log_prob_fn=self._like_func, random=self._random)

        logger.debug("Starting proposal loop...")

        for j in [0, 1]:

            # Get positions of walkers to be updated and walkers to be sampled.
            jupdate = j
            jsample = (j + 1) % 2
            pupdate = p[:, jupdate::2, :]   # current
            psample = p[:, jsample::2, :]   # complementary
            loglpup = logl[:, jupdate::2]   # current logl

            # propose new points
            logger.debug("Proposing...")
            qs, logf = self._proposal.propose(pupdate, psample, loglpup, model)

            # ensure parameters in bounds
            logger.debug("Moving in bounds...")
            qs  = np.array([list(self.mapf(eval_func_tuple,
                                           zip(repeat(apply_bounds), qi,
                                               repeat(self._period_reflect_list),
                                               repeat(self._bounds))))
                            for qi in qs])

            # evaluate likelihoods
            qslogl, qslogp  = self._evaluate(qs)
            qslogpost       = self._tempered_likelihood(qslogl) + qslogp

            # compute acceptance
            logger.debug("Compute acceptance...")
            accepts = logf + qslogpost - logpost[:, jupdate::2] > np.log(self._random.uniform(low=0.0, high=1.0, size=(self.ntemps,self.nwalkers//2)))
            accepts = accepts.flatten()

            # update samples
            logger.debug("Updating ensamle...")
            pupdate.reshape((-1, self.dim))[accepts, :]     = qs.reshape((-1, self.dim))[accepts, :]
            logpost[:, jupdate::2].reshape((-1,))[accepts]  = qslogpost.reshape((-1,))[accepts]
            logl[:, jupdate::2].reshape((-1,))[accepts]     = qslogl.reshape((-1,))[accepts]

            accepts = accepts.reshape((self.ntemps, self.nwalkers//2))

            self.nprop[:, jupdate::2] += 1.0
            self.nprop_accepted[:, jupdate::2] += accepts

        logger.debug("Samples proposed.")
        return p, logpost, logl

    def sample(self, iterations=1):

        # Set initial walker positions.
        p = self._p0

        # Check for dodgy inputs.
        if np.any(np.isinf(p)):
            logger.error("At least one parameter value was infinite.")
            raise ValueError("At least one parameter value was infinite.")
        if np.any(np.isnan(p)):
            logger.error("At least one parameter value was NaN.")
            raise ValueError("At least one parameter value was NaN.")

        # If we have no likelihood or prior values, compute them.
        if self._logposterior0 is None or self._loglikelihood0 is None:
            logl, logp = self._evaluate(p)
            logpost = self._tempered_likelihood(logl) + logp

            self._loglikelihood0 = logl
            self._logposterior0 = logpost
        else:
            logl = self._loglikelihood0
            logpost = self._logposterior0

        for i in range(iterations):

            # perform move
            logger.debug("Proposing samples in parallel")
            p, logpost, logl = self._propose(p, logpost, logl)
            logger.debug("Samples proposed")

            # perform swap
            logger.debug("Swapping samples in serial")
            p, ratios = self._temperature_swaps(self._betas, p, logpost, logl)
            logger.debug("Samples swapped")

            # update state and iterator
            self._save_history((p , logl, logpost))
            self._time += 1

        # store self properties
        self._p0 = p
        self._logposterior0 = logpost
        self._loglikelihood0 = logl

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the
        special case where beta == 0 *and* we're outside the likelihood support.

        Here, we find a singularity that demands more careful attention; we allow the likelihood to
        dominate the temperature, since wandering outside the likelihood support causes a discontinuity.

        """

        if betas is None:
            betas = self._betas
        betas = betas.reshape((-1, 1))

        with np.errstate(invalid='ignore'):
            loglT = logl * betas
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _temperature_swaps(self, betas, p, logpost, logl):
        """
        Perform parallel-tempering temperature swaps on the state
        in ``p`` with associated ``logpost`` and ``logl``.

        """
        ntemps = len(betas)
        ratios = np.zeros(ntemps - 1)

        for i in range(ntemps - 1, 0, -1):
            bi = betas[i]
            bi1 = betas[i - 1]

            dbeta = bi1 - bi

            iperm = self._random.permutation(self.nwalkers)
            i1perm = self._random.permutation(self.nwalkers)

            raccept = np.log(self._random.uniform(size=self.nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            self.nswap[i] += self.nwalkers
            self.nswap[i - 1] += self.nwalkers

            asel = (paccept > raccept)
            nacc = np.sum(asel)

            self.nswap_accepted[i] += nacc
            self.nswap_accepted[i - 1] += nacc

            ratios[i - 1] = nacc / self.nwalkers

            ptemp = np.copy(p[i, iperm[asel], :])
            logltemp = np.copy(logl[i, iperm[asel]])
            logprtemp = np.copy(logpost[i, iperm[asel]])

            p[i, iperm[asel], :] = p[i - 1, i1perm[asel], :]
            logl[i, iperm[asel]] = logl[i - 1, i1perm[asel]]
            logpost[i, iperm[asel]] = logpost[i - 1, i1perm[asel]] \
                - dbeta * logl[i - 1, i1perm[asel]]

            p[i - 1, i1perm[asel], :] = ptemp
            logl[i - 1, i1perm[asel]] = logltemp
            logpost[i - 1, i1perm[asel]] = logprtemp + dbeta * logltemp

        return p, ratios

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        """

        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def _expand_history(self, iterations):
        """
        Expand ``self._chain``, ``self._logposterior``,
        ``self._loglikelihood``, and ``self._beta_history``
        ahead of run to make room for new samples.

        :return ``isave``:
            Returns the index at which to begin inserting new entries.

        """

        logger.debug("Expanding chains ...")
        if len(self._chain_history) == 0:
            self._chain_history = np.zeros((self.ntemps, self.nwalkers, iterations, self.dim))
            self._logposterior_history = np.zeros((self.ntemps, self.nwalkers, iterations))
            self._loglikelihood_history = np.zeros((self.ntemps, self.nwalkers, iterations))
        else:
            self._chain_history = np.concatenate((self._chain_history, np.zeros((self.ntemps, self.nwalkers,iterations, self.dim))), axis=2)
            self._logposterior_history = np.concatenate((self._logposterior_history, np.zeros((self.ntemps,self.nwalkers,iterations))),axis=2)
            self._loglikelihood_history = np.concatenate((self._loglikelihood_history, np.zeros((self.ntemps, self.nwalkers, iterations))), axis=2)
        logger.debug("Chains expanded.")

    def _save_history(self, args):
        """
            Expand histories with current sample
        """

        logger.debug("Updating chains ...")
        p, logl, logp = args
        self._chain_history[:, :, self._time, :] = p
        self._logposterior_history[:, :, self._time] = logp
        self._loglikelihood_history[:, :, self._time] = logl

        self._acceptance_history.append(self.nprop_accepted / self.nprop)
        self._tswap_acceptance_history.append(self.nswap_accepted / self.nswap)
        logger.debug("Chains updated.")

    def _get_chain(self, nburn):
        """
        Get chain from history

        :param nburn:
            The number of burn-in iterations to be discarded.

        """
        self._chain         = self._chain_history[:,:,nburn:,:]
        self._logposterior  = self._logposterior_history[:,:,nburn:]
        self._loglikelihood = self._loglikelihood_history[:,:,nburn:]

    def log_evidence_estimate(self, logls=None):
        """
        Thermodynamic integration estimate of the evidence for the sampler.

        :param logls: (optional) The log-likelihoods to use for
            computing the thermodynamic evidence.  If ``None`` (the
            default), use the stored log-likelihoods in the sampler.
            Should be of shape ``(Ntemps, Nwalkers, Nsamples)``.

        :return ``(logZ, dlogZ)``: Returns an estimate of the
            log-evidence and the error associated with the finite
            number of temperatures at which the posterior has been
            sampled.

        For details, see ``thermodynamic_integration_log_evidence``.
        """

        if logls is None:
            if self.loglikelihood is not None:
                logls = self.loglikelihood
            else:
                raise ValueError('No log likelihood values available.')

        mean_logls = np.mean(np.mean(logls, axis=1)[:, 0:], axis=1)
        return thermodynamic_integration_log_evidence(self._betas, mean_logls)

    @property
    def random(self):
        """
        Returns the random number generator for the sampler.

        """

        return self._random

    @property
    def betas(self):
        """
        Returns the current inverse temperature ladder of the sampler.

        """
        return self._betas

    @property
    def time(self):
        """
        Returns the current time, in iterations, of the sampler.

        """
        return self._time

    @property
    def chain(self):
        """
        Returns the stored chain of samples; shape ``(Ntemps,
        Nwalkers, Nsteps, Ndim)``.

        """
        return self._chain

    @property
    def flatchain(self):
        """Returns the stored chain, but flattened along the walker axis, so
        of shape ``(Ntemps, Nwalkers*Nsteps, Ndim)``.

        """

        s = self.chain.shape

        return self._chain.reshape((s[0], -1, s[3]))

    @property
    def logprobability(self):
        """
        Matrix of logprobability values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._logposterior

    @property
    def loglikelihood(self):
        """
        Matrix of log-likelihood values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._loglikelihood

    @property
    def tswap_acceptance_fraction(self):
        """
        Returns an array of accepted temperature swap fractions for
        each temperature; shape ``(ntemps, )``.

        """
        return self.nswap_accepted / self.nswap

    @property
    def ntemps(self):
        """
        The number of temperature chains.

        """
        return len(self._betas)

    @property
    def acceptance_fraction(self):
        """
        Matrix of shape ``(Ntemps, Nwalkers)`` detailing the
        acceptance fraction for each walker.

        """
        return self.nprop_accepted / self.nprop

    @property
    def acor(self):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        """
        return self.get_autocorr_time()

    @property
    def untemp_index(self):
        """
            Returns a matrix of autocorrelation lengths for each
            parameter in each temperature of shape ``(Ntemps, Ndim)``.

            """
        return int(np.where(self._betas==1.)[0])

    def get_autocorr_time(self, window=50):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        :param window: (optional)
            The size of the windowing function. This is equivalent to the
            maximum number of lags to use. (default: 50)

        """
        acors = np.zeros((self.ntemps, self.dim))

        for i in range(self.ntemps):
            x = np.mean(self._chain[i, :, :, :], axis=0)
            acors[i, :] = autocorr_integrated_time(x, window=window)
        return acors

    def get_autocorr_time_untemp(self, nburn, window=50):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        :param window: (optional)
            The size of the windowing function. This is equivalent to the
            maximum number of lags to use. (default: 50)

        """
        x = np.mean(self._chain_history[self.untemp_index, :, nburn:, :], axis=0)
        return autocorr_integrated_time(x, window=window)

class SamplerPTMCMC(SamplerBody):

    def __initialize__(self, posterior, nwalk, ntemps, tmax=None,
                       nburn=10000, nout=8000,
                       proposals=None,
                       proposals_kwargs={'use_slice': False, 'use_gw': False},
                       pool=None, **kwargs):

        # initialize PTMCMC parameters
        self.nburn          = nburn
        self.nout           = nout

        if nwalk < 2*len(posterior.prior.names):
            logger.warning("Requested number of walkers < 2 * Ndim. This may generate problems in the exploration of the parameters spaces.")

        if ntemps < 2*np.sqrt(len(posterior.prior.names)):
            logger.warning("Requested number of temperature < 2 * sqrt(Ndim). This may generate problems in the exploration of the parameters spaces.")

        # initialize proposals
        if proposals == None:
            logger.info("Initializing proposal methods ...")
            proposals = initialize_proposals(posterior.like, posterior.prior, **proposals_kwargs)

        # if tmax is not specified, use inf (beta min = 0)
        if tmax == None:
            tmax = np.inf

        # initialize ptmcmc sampler
        self.sampler = _PTMCMC(nwalkers    = nwalk,
                               ntemps      = ntemps,
                               Tmax        = tmax,
                               posterior   = posterior,
                               proposal    = proposals,
                               pool        = pool)

        # extract prior samples for initial state
        logger.info("Extracting prior samples ...")
        self.sampler._p0 = np.array([posterior.prior.get_prior_samples(nwalk) for _ in range(self.sampler.ntemps)])

        # initialize other variables
        self.neff       = '(burn-in)'
        self.acl        = '(burn-in)'
        self.stop       = False

    def __restore__(self, pool, **kwargs):

        # re-initialize pool
        self.sampler.pool = pool

        # re-initialize mapf
        if pool == None:
            self.sampler.mapf = map
        else:
            self.sampler.mapf = self.sampler.pool.map

        # re-initialize seed
        self.sampler._random = np.random

    def __run__(self):

        # run the chains
        while not self.stop:

            # expand history chains
            self.sampler._expand_history(iterations=self.ncheckpoint)

            # make steps
            self.sampler.sample(iterations=self.ncheckpoint)

            # update sampler status
            self.update()

            # compute stopping condition
            self._stop_sampler()

        # final store inference
        self.store()

    def __update__(self):

        logger.debug("Setting updated printings...")

        # compute acceptance
        acc_T0  = (self.sampler.nprop_accepted[self.sampler.untemp_index]).sum()/self.sampler.nwalkers/self.sampler._time
        #acc_all = (self.sampler.nprop_accepted).sum()/self.sampler.nwalkers/self.sampler.ntemps/self.sampler._time
        swp_acc = (self.sampler.nswap_accepted/self.sampler.nswap).sum()/self.sampler.ntemps

        # compute logLs
        this_logL       = np.array(self.sampler._loglikelihood0[self.sampler.untemp_index])
        logL_mean       = logsumexp(this_logL) - np.log(self.sampler.nwalkers)
        logL_max        = np.max(this_logL)

        args    = { 'it' : self.sampler._time,
                    'acc' : '{:.2f}%'.format(acc_T0*100.),
                    'swp' : '{:.2f}%'.format(swp_acc*100.),
                    'acl' : 'n/a',
                    'stat' : 'burn-in',
                    'logLmean' : '{:.2f}'.format(logL_mean),
                    'logLmax' : '{:.2f}'.format(logL_max)
                    }

        # update logger
        if not isinstance(self.neff, str):
            args['acl']     = self.acl
            args['stat']    = '{:.2f}%'.format(self.neff*100/self.nout)

        return args

    def _stop_sampler(self):

        # if it > nburn, compute acl every 100 iterations
        # obs. we have to include some other iteration in order to
        # collect a sufficiently large set of sample for ACL estimation
        logger.debug("Evaluating stopping condition...")
        if (self.sampler._time > self.nburn+self.ncheckpoint):

            # compute ACL of untempered chain
            acls = self.sampler.get_autocorr_time_untemp(nburn=self.nburn)

            try:
                self.acl    = int(np.max([ ai for ai in acls if not np.isnan(ai)]))
                self.neff   = (self.sampler._time-self.nburn)*self.sampler.nwalkers//self.acl
            except Exception:
                self.acl    = np.inf
                self.neff   = 0

            # if the number of collected samples (in the T=1 chain)
            # is greater than nout, the sampling is done
            if self.neff >= self.nout :
                self.stop = True
        logger.debug("Stopping condition evaluated.")

    def get_posterior(self):

        # update chains
        self.sampler._get_chain(int(self.nburn))

        s = self.sampler._chain.shape
        self.outchain = self.sampler._chain.reshape((s[0], -1, s[3])) # (ntemps, nwalk*nsteps , ndim)

        # extract posterior only from chain at T=1
        self.posterior_samples  = self.outchain[self.sampler.untemp_index][::self.acl]

        # extract logL for posterior samples,
        # log-prior is computed when the posterior file is saved
        logL  = self.sampler._loglikelihood.reshape((s[0],-1))[self.sampler.untemp_index][::self.acl]
        logP  = self.sampler._logposterior.reshape((s[0],-1))[self.sampler.untemp_index][::self.acl]
        logpr = logP - logL

        logger.info("  - autocorr length : {}".format(self.acl))

        self.real_nout = self.posterior_samples.shape[0]
        logger.info("  - number of posterior samples : {}".format(self.real_nout))

        post_file = open(self.outdir + '/posterior.dat', 'w')

        post_file.write('#')
        for n in range(self.sampler.dim):
            post_file.write('{}\t'.format(self.names[n]))
        post_file.write('logL\t logPrior\n')

        for i in range(self.real_nout):
            for j in range(self.sampler.dim):
                post_file.write('{}\t'.format(self.posterior_samples[i][j]))
            post_file.write('{}\t{}\n'.format(logL[i],logpr[i]))

        post_file.close()

        # estimate evidence
        logz , logzerr  = self.sampler.log_evidence_estimate()
        evidence_file = open(self.outdir + '/evidence.dat', 'w')
        evidence_file.write('betas  = {}\n'.format(self.sampler._betas))
        evidence_file.write('logZ   = {} +/- {}\n'.format(logz,logzerr))
        evidence_file.close()

    def make_plots(self):

        try:
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning("Impossible to produce standard plots. Cannot import matplotlib.")

        # create plot directory
        plot_dir = os.path.join(self.outdir,'ptmcmc')
        ensure_dir(plot_dir)

        try:

            # plot sample chains
            chain_samples  = np.transpose(self.outchain[self.sampler.untemp_index])
            for i,ni in enumerate(self.names):

                fig = plt.figure()
                plt.scatter(np.arange(len(chain_samples[i])), chain_samples[i], marker='.')
                # plt.axvline(self.nburn*self.sampler.nwalkers, color='k', ls='--', label='burn-in')
                plt.ylabel(ni)
                plt.xlabel('iteration')

                plt.legend()
                plt.savefig(plot_dir+'/chain_{}.png'.format(ni), dpi=200)
                plt.close()

            # plot logL chains
            for i in range(self.sampler.ntemps):

                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax1.plot(self.sampler._loglikelihood[i], lw=0.3)
                ax2.plot(self.sampler._logposterior[i], lw=0.3)

                ax1.set_title("T={:.5f}".format(1./self.sampler._betas[i]))
                ax1.set_ylabel('lnL')
                ax1.set_xticks([])

                ax2.set_ylabel('lnP')
                ax2.set_xlabel('iteration')

                plt.subplots_adjust(hspace=0.)
                plt.savefig(plot_dir+'/chain_logP_{}.png'.format(i), dpi=200)

                plt.close()

        except Exception:
            pass

        try:

            import corner

            corner.corner(self.posterior_samples,
                          smooth=True, smooth1d=True, labels=self.names,
                          quantiles=[0.50,0.90])

            plt.savefig(plot_dir+'/corner.png'.format(i), dpi=200)
            plt.close()

        except Exception:
            pass

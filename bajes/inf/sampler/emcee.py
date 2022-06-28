from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from itertools import repeat

import emcee
from emcee import State
from emcee.model import Model

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from . import SamplerBody
from .proposal import _init_proposal_methods
from ..utils import apply_bounds
from ...pipe import eval_func_tuple

def accept_func(new_logp, old_logp, fact):
    if (fact + new_logp - old_logp) > np.log(np.random.rand()):
        return True
    else:
        return False

def walkers_independent(coords):
    if not np.all(np.isfinite(coords)):
        return False
    C = coords - np.mean(coords, axis=0)[None, :]
    C_colmax = np.amax(np.abs(C), axis=0)
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C ** 2, axis=0))
    C /= C_colsum
    return np.linalg.cond(C.astype(float)) <= 1e8

def walkers_independent_cov(coords):
    C = np.cov(coords, rowvar=False)
    if np.any(np.isnan(C)):
        return False
    return _scaled_cond(np.atleast_2d(C)) <= 1e8

def _scaled_cond(a):
    asum = np.sqrt((a ** 2).sum(axis=0))[None, :]
    if np.any(asum == 0):
        return np.inf
    b = a / asum
    bsum = np.sqrt((b ** 2).sum(axis=1))[:, None]
    if np.any(bsum == 0):
        return np.inf
    c = b / bsum
    return np.linalg.cond(c.astype(float))

def initialize_proposals(like, priors, use_slice=False, use_gw=False):

    props = {'eig': 25., 'dif': 25., 'str': 20., 'wlk': 15., 'kde': 10., 'pri': 5.}
    prop_kwargs  = {}

    if use_slice:
        props['slc'] = 30.

    if use_gw:
        props['gwt'] = 20.
        prop_kwargs['like'] = like
        prop_kwargs['dets'] = like.dets

    return BajesMCMCProposal(priors, props, **prop_kwargs)

class BajesMCMCProposal(emcee.moves.red_blue.RedBlueMove):

    def __init__(self, priors, props=None,
                 nsplits=2, randomize_split=True,
                 **kwargs):
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
            - nsplits  : number of splits for complementary ensemble (default 2)
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

        # run emcee proposal init
        super(BajesMCMCProposal, self).__init__(nsplits=nsplits, randomize_split=randomize_split)

    def get_proposal(self, s, c, p, model):
        _p = model.random.choice(self._proposals, p=self._weights)
        return _p.get_proposal(s, c, p, model)

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance
            Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
            subset of walkers.
            random: A numpy-compatible random number state.
            """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError("It is unadvisable to use a proposal move "
                               "with fewer walkers than twice the number of "
                               "dimensions.")

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits

        if self.randomize_split:
            model.random.shuffle(inds)

        for split in range(self.nsplits):
            S1 = inds == split

            # Select one subset of the ensamble
            sets        = [state.coords[inds == j] for j in range(self.nsplits)]
            logp_sets   = [state.log_prob[inds == j] for j in range(self.nsplits)]

            # Assign varibles for set and complementary
            s       = sets[split]
            c       = sets[:split] + sets[split + 1 :]
            probs   = logp_sets[split]

            # Get the move-specific proposal
            q, factors  = self.get_proposal(s, c, probs, model)

            # apply periodic/reflective bounds
            qinbounds   = np.array(list(model.map_fn(eval_func_tuple, zip(repeat(apply_bounds), q, repeat(self.period_reflect_list), repeat(self.bounds)))))

            # check NaNs
            isnan       = np.isnan(qinbounds)
            if np.sum(isnan):
                raise ValueError("{}\n{}\nAt least one parameter was NaN in {}.".format(qinbounds, self.names, self))

            # Compute the lnprobs of the proposed position.
            new_log_probs, _ = model.compute_log_prob_fn(qinbounds)

            # Loop over the walkers and update them accordingly.
            new_acc     = np.array(list(map(accept_func, new_log_probs, state.log_prob[all_inds[S1]], factors)))
            accepted[all_inds[S1]] = new_acc

            # Renew running chain
            new_state = State(qinbounds, log_prob=new_log_probs)
            state = self.update(state, new_state, accepted, S1)

        return state, accepted

class _EnsembleSampler(emcee.EnsembleSampler):
    """
        Wrapper for emcee.EnsembleSampler with corrected __getstate__ method
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        state["pool"] = None
        return state

class SamplerMCMC(SamplerBody):

    def __initialize__(self, posterior, nwalk,
                       nburn=10000, nout=2000,
                       proposals=None, proposals_kwargs={'use_slice': False, 'use_gw': False},
                       pool=None, **kwargs):

        # initialize MCMC parameters
        self.nburn  = nburn
        self.nout   = nout

        # warnings
        if nwalk < posterior.prior.ndim**2:
            logger.warning("Requested number of walkers < Ndim^2. This may generate problems in the exploration of the parameters space.")

        # inilialize backend
        from emcee.backends import Backend
        backend = Backend()

        # initialize proposals
        if proposals == None:
            logger.info("Initializing proposal methods ...")
            proposals = initialize_proposals(posterior.like, posterior.prior, **proposals_kwargs)

        # initialize sampler
        logger.info("Initializing sampler ...")
        self.sampler = _EnsembleSampler(nwalkers    = nwalk,
                                        ndim        = posterior.prior.ndim,
                                        log_prob_fn = posterior.log_post,
                                        moves       = proposals,
                                        backend     = backend,
                                        pool        = pool)

        # avoid default emcee wrapper
        self.sampler.log_prob_fn = posterior.log_post

        # extract prior samples for initial state
        logger.info("Extracting prior samples ...")
        self._previous_state = posterior.prior.get_prior_samples(nwalk)

        # initialize other variables
        self.neff   = '(burn-in)'
        self.acl    = '(burn-in)'
        self.stop   = False

    def __restore__(self, pool, **kwargs):

        # re-initialize pool
        if pool == None:
            self.sampler.pool   = pool
        else:
            self.sampler.pool   = pool

    def __update__(self):

        # compute acceptance
        acc = (self.sampler.backend.accepted).sum()/self.sampler.nwalkers/self.sampler.backend.iteration

        # compute logLs
        this_logL   = np.array(self.sampler.backend.get_last_sample().log_prob)
        logL_mean   = logsumexp(this_logL) - np.log(self.sampler.nwalkers)
        logL_max    = np.max(this_logL)

        args    = { 'it' : self.sampler.backend.iteration,
                    'acc' : '{:.2f}%'.format(acc*100.),
                    'acl' : 'n/a',
                    'stat' : 'burn-in',
                    'logPmean' : '{:.2f}'.format(logL_mean),
                    'logPmax' : '{:.2f}'.format(logL_max)
                    }

        # update logger
        if not isinstance(self.neff, str):
            args['acl']     = self.acl
            args['stat']    = '{:.2f}%'.format(self.neff*100/self.nout)

        return args

    def _stop_sampler(self):

        # if it > nburn, compute acl every step
        if (self.sampler.backend.iteration > self.nburn):

            acls        = self.sampler.backend.get_autocorr_time(discard=self.nburn, quiet=True, tol=2)
            acls_clean  = [ai for ai in acls if not np.isnan(ai)]

            if len(acls_clean) == 0:
                logger.warning("ACL was NaN for all parameters.")
                self.acl    = np.inf
                self.neff   = 0
            else:
                self.acl    = int(np.max(acls_clean))
                if self.acl < 1: self.acl = 1
                self.neff   = (self.backend.iteration-self.nburn)*self.nwalkers//self.acl

            # if the number of collected samples is greater than nout, the sampling is done
            if self.neff >= self.nout :
                self.stop = True

    def __run__(self):

        while not self.stop:

            # make steps
            for results in self.sampler.sample(self._previous_state, iterations=self.ncheckpoint, tune=True):
                pass

            # update previous state
            self._previous_state  = results

            # update sampler status
            self.update()

            # compute stopping condition
            self._stop_sampler()

        # final store inference
        self.store()

    def get_posterior(self):

        try:
            acls = self.sampler.backend.get_autocorr_time(discard=self.nburn, quiet=True, tol=2)
            acl = int (np.max([ ai for ai in acls if (not np.isnan(ai) and not np.isinf(ai))]))

        except Exception:
            # if backend.get_autocorr_time does not work, the correlation of the samples is too high,
            # or some parameters were NaN or Inf. Then acl is fixed to 1 and nburn is fixed to nstep//2.
            # However, this should not happen. If this happen, this could be a problem of the proposals.
            # Please report the error.
            acl = 1
            self.nburn = np.abs(self.sampler.backend.iteration)//2
            logger.warning("Warning: NaN or Inf occurred in ACL computation. ACL will be set equal to 1 and nburn will be fixed to nstep//2. Try increasing the number of walkers for the MCMC exploration if you want to improve your sampling.")

        self.posterior_samples  = np.array(self.sampler.backend.get_chain(flat=True, discard=self.nburn, thin=acl))
        logP                    = np.array(self.sampler.backend.get_log_prob(flat=True, discard=self.nburn, thin=acl))

        logger.info("  - autocorr length : {}".format(acl))

        self.real_nout = self.posterior_samples.shape[0]
        logger.info("  - number of posterior samples : {}".format(self.real_nout))

        post_file = open(self.outdir + '/posterior.dat', 'w')

        post_file.write('#')
        for n in range(self.ndim):
            post_file.write('{}\t'.format(self.names[n]))
        post_file.write('logP\n')

        for i in range(self.real_nout):
            for j in range(self.ndim):
                post_file.write('{}\t'.format(self.posterior_samples[i][j]))
            post_file.write('{}\n'.format(logP[i]))

        post_file.close()

    def make_plots(self):

        try:
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning("Impossible to produce standard plots. Cannot import matplotlib.")

        try:

            chain_prob = np.array(self.backend.get_log_prob(flat=False, discard=0, thin=1))

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(chain_prob, lw=0.3)
            ax1.set_ylabel('lnP - lnZnoise')
            ax1.set_xlabel('iteration')

            plt.savefig(self.outdir+'/chain_logP.png', dpi=200)

            plt.close()

        except Exception:
            pass

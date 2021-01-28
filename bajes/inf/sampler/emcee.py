from __future__ import division, unicode_literals, absolute_import
import numpy as np
import os
import signal
import tracemalloc

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

from .proposal import _init_proposal_methods
from ..utils import apply_bounds
from ...pipe import erase_init_wrapper, data_container, eval_func_tuple, display_memory_usage

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

class SamplerMCMC(emcee.EnsembleSampler):

    def __init__(self, posterior, nwalk,
                 nburn=10000, nout=2000,
                 proposals=None,
                 proposals_kwargs={'use_slice': False, 'use_gw': False},
                 pool=None, ncheckpoint=0,
                 outdir='./', resume='/resume.pkl',
                 seed=None, **kwargs):
        
        self.resume = resume
        self.outdir = outdir
        
        # restore inference from existing container
        if os.path.exists(self.outdir + self.resume):
            self.restore_inference(pool)
        
            # update nout and nburn
            self.nburn  = nburn
            self.nout   = nout

        # initialize a new inference
        else:
            
            # initialize signal handler
            try:
                signal.signal(signal.SIGTERM, self.store_inference_and_exit)
                signal.signal(signal.SIGINT, self.store_inference_and_exit)
                signal.signal(signal.SIGALRM, self.store_inference_and_exit)
            except AttributeError:
                logger.warning("Impossible to set signal attributes.")

            # initialize MCMC parameters
            self.nburn  = nburn
            self.nout   = nout
            self.names  = posterior.prior.names
            
            if ncheckpoint == 0:
                # disable resume
                logger.info("Disabling checkpoint ...")
                self.ncheckpoint = 100 # print step
                self.store_flag = False
            else:
                # enable resume
                logger.info("Enabling checkpoint ...")
                self.ncheckpoint = ncheckpoint
                self.store_flag = True
            
            if nwalk < posterior.prior.ndim**2:
                logger.warning("Requested number of walkers < Ndim^2. This may generate problems in the exploration of the parameters space.")
                    
            # initialize random seed
            if seed == None:
                import time
                self.seed = int(time.time())
            else:
                self.seed = seed
            np.random.seed(self.seed)
            
            # inilialize backend
            from emcee.backends import Backend
            backend = Backend()
            
            # initialize proposals
            if proposals == None:
                logger.info("Initializing proposal methods ...")
                proposals = initialize_proposals(posterior.like, posterior.prior, **proposals_kwargs)

            # initialize emcee sampler
            super(SamplerMCMC, self).__init__(nwalkers    = nwalk,
                                              ndim        = posterior.prior.ndim,
                                              log_prob_fn = posterior.log_post,
                                              moves       = proposals,
                                              backend     = backend,
                                              pool        = pool)
                                                  
            # avoid automatic function wrapper of emcee
            self.log_prob_fn = posterior.log_post
            self._propose    = self._moves[0].propose

            # extract prior samples for initial state
            logger.info("Extracting prior samples ...")
            self._previous_state = posterior.prior.get_prior_samples(nwalk)

            # initialize other variables
            self.neff   = '(burn-in)'
            self.acl    = '(burn-in)'
            self.stop   = False

    def __getstate__(self):
        # get __dict__ of parent+current class
        full_dict  = self.__dict__.copy()
        # remove not picklable objects
        if 'pool' in list(full_dict.keys()):
            full_dict['pool'] = None
        return full_dict

    def restore_inference(self, pool):

        # extract container
        logger.info("Restoring inference from existing container ...")
        dc                  = data_container(self.outdir + self.resume)
        container           = dc.load()
        
        # sampler check
        if container.tag != 'mcmc':
            logger.error("Container carries a {} inference, while MCMC was requested.".format(container.tag.upper()))
            raise AttributeError("Container carries a {} inference, while MCMC was requested.".format(container.tag.upper()))
        
        previous_inference  = container.inference

        # extract previous variables and methods
        for kw in list(previous_inference.keys()):
            self.__setattr__(kw, previous_inference[kw])
        
        # re-initialize signal
        try:
            signal.signal(signal.SIGTERM,   self.store_inference_and_exit)
            signal.signal(signal.SIGINT,    self.store_inference_and_exit)
            signal.signal(signal.SIGALRM,   self.store_inference_and_exit)
        except AttributeError:
            logger.warning("Impossible to set signal attributes.")

        # re-initialize seed
        np.random.seed(self.seed)    
        
        # re-initialize emcee sampler
        # i.e. all the parameters are imported from previous sampler,
        # then the previous iterations are restored (in backend)
        self.temp_log_prob_fn = self.log_prob_fn
        super(SamplerMCMC, self).__init__(nwalkers    = self.nwalkers,
                                            ndim        = self.ndim,
                                            log_prob_fn = self.log_prob_fn,
                                            moves       = np.transpose([self._moves, self._weights]),
                                            backend     = self.backend,
                                            pool        = pool)

        # avoid automatic function wrapper of emcee
        self.log_prob_fn = self.temp_log_prob_fn
        self._propose    = self._moves[0].propose

    def store_inference_and_exit(self, signum=None, frame=None):
        # exit function when signal is revealed
        logger.info("Run interrupted by signal {}, checkpoint and exit.".format(signum))
        os._exit(signum)

    def store_inference(self):
        # save inference in pickle file
        dc = data_container(self.outdir+self.resume)
        dc.store('tag', 'mcmc')
        dc.store('inference', self.__getstate__())
        dc.save()

    def run(self):
        # run the chains
        logger.info("Running {} walkers ...".format(self.nwalkers))
        self.run_mcmc()

    def update_sampler(self):
    
        # compute acceptance
        acc = np.sum(self.backend.accepted)/self.nwalkers/self.backend.iteration
        
        # compute logLs
        this_logL   = np.array(self.backend.get_last_sample().log_prob)
        logL_mean   = logsumexp(this_logL) - np.log(self.nwalkers)
        logL_max    = np.max(this_logL)

        # store inference
        if self.store_flag:
            self.store_inference()

        # update logger
        if isinstance(self.neff, str):
            logger.info(" - it : {:d} - stat : {} - acl : N/A - acc : {:.3f} - logPmean : {:.5g} - logPmax : {:.5g}".format(self.backend.iteration,
                                                                                                                                  self.neff, acc, logL_mean,logL_max))
        else:
            logger.info(" - it : {:d} - stat : {:.3f}% - acl : {} - acc : {:.3f} - logPmean : {:.5g} - logPmax : {:.5g}".format(self.backend.iteration,
                                                                                                                                      self.neff*100/self.nout,self.acl,
                                                                                                                                      acc,logL_mean,logL_max))

    def stop_sampler(self):
    
        # if it > nburn, compute acl every step
        if (self.backend.iteration > self.nburn):
        
            acls        = self.backend.get_autocorr_time(discard=self.nburn, quiet=True, tol=2)
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

    def sample(
        self,
        initial_state,
        log_prob0=None,  # Deprecated
        rstate0=None,  # Deprecated
        blobs0=None,  # Deprecated
        iterations=1,
        tune=False,
        skip_initial_state_check=False,
        thin_by=1,
        thin=None,
        store=True,
        progress=False,
    ):
        """
            emcee sample method
        """
        # Interpret the input as a walker state and check the dimensions.
        state = State(initial_state, copy=True)
        if np.shape(state.coords) != (self.nwalkers, self.ndim):
            raise ValueError("incompatible input dimensions")
        if (not skip_initial_state_check) and (
            not walkers_independent(state.coords)
        ):
            raise ValueError(
                "Initial state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        if rstate0 is not None:
            deprecation_warning(
                "The 'rstate0' argument is deprecated, use a 'State' "
                "instead"
            )
            state.random_state = rstate0
        self.random_state = state.random_state

        # If the initial log-probabilities were not provided, calculate them
        # now.
        if log_prob0 is not None:
            deprecation_warning(
                "The 'log_prob0' argument is deprecated, use a 'State' "
                "instead"
            )
            state.log_prob = log_prob0
        if blobs0 is not None:
            deprecation_warning(
                "The 'blobs0' argument is deprecated, use a 'State' instead"
            )
            state.blobs = blobs0
        if state.log_prob is None:
            state.log_prob, state.blobs = self.compute_log_prob(state.coords)
        if np.shape(state.log_prob) != (self.nwalkers,):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(state.log_prob)):
            raise ValueError("The initial log_prob was NaN")

        # Deal with deprecated thin argument
        if thin is not None:
            deprecation_warning(
                "The 'thin' argument is deprecated. " "Use 'thin_by' instead."
            )

            # Check that the thin keyword is reasonable.
            thin = int(thin)
            if thin <= 0:
                raise ValueError("Invalid thinning argument")

            yield_step = 1
            checkpoint_step = thin
            iterations = int(iterations)
            if store:
                nsaves = iterations // checkpoint_step
                self.backend.grow(nsaves, state.blobs)

        else:
            # Check that the thin keyword is reasonable.
            thin_by = int(thin_by)
            if thin_by <= 0:
                raise ValueError("Invalid thinning argument")

            yield_step = thin_by
            checkpoint_step = thin_by
            iterations = int(iterations)
            if store:
                self.backend.grow(iterations, state.blobs)

        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map
        model = Model(
            self.log_prob_fn, self.compute_log_prob, map_fn, self._random
        )

        # Inject the progress bar
        total = iterations * yield_step
        i = 0
        for _ in range(iterations):
            for _ in range(yield_step):

                # Propose
                state, accepted = self._propose(model, state)
                state.random_state = self.random_state

                # if tune:
                #   move.tune(state, accepted)

                # Save the new step
                if store and (i + 1) % checkpoint_step == 0:
                    self.backend.save_step(state, accepted)
                
                i += 1

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            yield state

    def run_mcmc(self):
    
        while not self.stop:
            
            # make steps
            for results in self.sample(self._previous_state, iterations=self.ncheckpoint, tune=True):
                pass

            # update previous state
            self._previous_state  = results

            # update sampler status
            self.update_sampler()

            # compute stopping condition
            self.stop_sampler()
    
            if tracemalloc.is_tracing():
                display_memory_usage(tracemalloc.take_snapshot())
                tracemalloc.clear_traces()

        # final store inference
        self.store_inference()

    def get_posterior(self):
        
        try:
            acls = self.backend.get_autocorr_time(discard=self.nburn, quiet=True, tol=2)
            acl = int (np.max([ ai for ai in acls if (not np.isnan(ai) and not np.isinf(ai))]))

        except Exception:
            # if backend.get_autocorr_time does not work, the correlation of the samples is too high,
            # or some parameters were NaN or Inf. Then acl is fixed to 1 and nburn is fixed to nstep//2.
            # However, this should not happen. If this happen, this could be a problem of the proposals.
            # Please report the error.
            acl = 1
            self.nburn = np.abs(self.backend.iteration)//2
            logger.warning("Warning: NaN or Inf occurred in ACL computation. ACL will be set equal to 1 and nburn will be fixed to nstep//2. Try increasing the number of walkers for the MCMC exploration if you want to improve your sampling.")

        self.posterior_samples  = np.array(self.backend.get_chain(flat=True, discard=self.nburn, thin=acl))
        logP                    = np.array(self.backend.get_log_prob(flat=True, discard=self.nburn, thin=acl))
        
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





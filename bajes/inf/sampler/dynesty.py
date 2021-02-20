from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

import dynesty
from dynesty.utils import unitcheck

from . import SamplerBody
from ..utils import estimate_nmcmc, list_2_dict

def void():
    pass

def reflect(u):
    idxs_even = np.mod(u, 2) < 1
    u[idxs_even] = np.mod(u[idxs_even], 1)
    u[~idxs_even] = 1 - np.mod(u[~idxs_even], 1)
    return u

def initialize_proposals(maxmcmc, minmcmc, nact):
    # initialize proposals
    return BajesDynestyProposal(maxmcmc, walks=minmcmc, nact=nact)

def resample(samples, weights):
    
    if abs(np.sum(weights) - 1.) > 1e-30:
        # Guarantee that the weights will sum to 1.
        weights = np.array(weights) / np.sum(weights)
    
    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (np.random.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = []
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx.append(j)
            i += 1
        else:
            j += 1

    idx = np.array(idx,  dtype=np.int)
    return samples[idx]

def get_prior_samples_dynesty(nlive, ndim, like_fn, ptform_fn):
    
    n = 0
    u = []
    v = []
    logl = []
    
    while n < nlive:
        _u = np.random.uniform(0,1,ndim)
        _v = ptform_fn(_u)
        _l = like_fn(_v)
        if not np.isinf(_l):
            u.append(_u)
            v.append(_v)
            logl.append(_l)
            n +=1

    return [np.array(u), np.array(v), np.array(logl)]

class SamplerDynesty(SamplerBody):
    
    def __initialize__(self, posterior,
                       nlive, tolerance=0.1,
                       # bounding
                       bound_method='multi', vol_check=8., vol_dec=0.5,
                       # update
                       bootstrap=0, enlarge=1.5, facc=0.5, update_interval=None,
                       # proposal
                       proposals=None, nact = 5., maxmcmc=4096, minmcmc=32,
                       # first update
                       first_min_ncall = None, first_min_eff = 10,
                       # parallelization
                       nprocs=None, pool=None,
                       **kwargs):
        

        # initialize nested parameters
        self.nlive          = nlive
        self.tol            = tolerance
            
        # auxiliary arguments
        self.log_prior_fn = posterior.log_prior
            
        # warnings
        if self.nlive < self.ndim*(self.ndim-1)//2:
            logger.warning("Given number of live points < Ndim*(Ndim-1)/2. This may generate problems in the exploration of the parameters space.")

        # set up periodic and reflective boundaries
        periodic_inds   = np.concatenate(np.where(np.array(posterior.prior.periodics) == 1))
        reflective_inds = np.concatenate(np.where(np.array(posterior.prior.periodics) == 0))

        # initialize proposals
        if proposals == None:
            logger.info("Initializing proposal methods ...")
            proposals = initialize_proposals(maxmcmc, minmcmc, nact)

        # check additional sampler arguments
        if first_min_ncall == None:
            first_min_ncall = 2 * nlive
        if nprocs == None:
            nprocs = 1

        # initialize keyword args for dynesty
        sampler_kwargs  = { 'ndim':         self.ndim,
                            'nlive':        nlive,
                            'bound':        bound_method,
                            'sample':       'rwalk',
                            'periodic':     periodic_inds,
                            'reflective':   reflective_inds,
                            'facc':         facc,
                            'vol_check':    vol_check,
                            'vol_dec':      vol_dec,
                            'walks':        minmcmc,
                            'enlarge':      enlarge,
                            'bootstrap':    bootstrap,
                            'pool':         pool,
                            'queue_size':   max(nprocs-1,1),
                            'update_interval': update_interval,
                            'first_update': {'min_ncall':first_min_ncall, 'min_eff': first_min_eff},
                            'use_pool':     {'prior_transform': True,'loglikelihood': True, 'propose_point': True,'update_bound': True}
                            }

        like_fn         = posterior.log_like
        ptform_fn       = posterior.prior_transform
        self.sampler    = self._initialize_sampler(like_fn, ptform_fn, sampler_kwargs)

        # clean up sampler
        del self.sampler.cite
        del self.sampler.kwargs['cite']
        self.sampler.rstate = np.random

        # set proposal
        self.sampler.evolve_point = proposals.propose

    def _initialize_sampler(self, like_fn, ptform_fn, kwargs):
        # extract prior samples, ensuring finite logL
        logger.info("Extracting prior samples ...")
        live_points = get_prior_samples_dynesty(kwargs['nlive'], kwargs['ndim'], like_fn, ptform_fn)
        kwargs['live_points'] = live_points
        
        # initialize dynesty sampler
        logger.info("Initializing nested sampler ...")
        sampler = dynesty.NestedSampler(loglikelihood=like_fn, prior_transform=ptform_fn, **kwargs)
        del sampler._PROPOSE
        del sampler._UPDATE
        return sampler
            
    def __restore__(self, pool, **kwargs):

        # re-initialize pool
        if pool == None:
            self.sampler.pool   = pool
            self.sampler.M      = map
        else:
            self.sampler.pool   = pool
            self.sampler.M      = pool.map

        # re-initialize random
        self.sampler.rstate = np.random

    def __run__(self):
        
        # run the sampler
        for results in self.sampler.sample(dlogz=self.tol,save_samples=True,add_live=False):
            
            if self.sampler.it%self.ncheckpoint==0:
                self._last_iter = results
                self.update()

        # add live points to nested samples
        logger.info("Adding live points in nested samples ...")
        self.sampler.add_final_live(print_progress=False)

        # final store inference
        self.store()

    def __update__(self):

        (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = self._last_iter
        
        args = {'it' :      self.sampler.it,
                'eff' :     '{:.2f}%'.format(eff),
                'nc' :      nc,
                #'logL' :    '{:.3f}'.format(loglstar),
                'logLmax' : '{:.3f}'.format(np.max(self.sampler.live_logl)),
                'logZ' :    '{:.3f}'.format(logz),
                'H' :       '{:.2f}'.format(h),
                'dZ' :      '{:.3f}'.format(delta_logz)}
        
        return args

    def get_posterior(self):
        
        self.results            = self.sampler.results
        self.nested_samples     = self.results.samples
        logger.info(" - number of nested samples : {}".format(len(self.nested_samples)))
        
        # extract posteriors
        ns = []
        wt = []
        scale = np.array(self.sampler.saved_scale)
        for i in range(len(self.nested_samples)):
            
            this_params = list_2_dict(self.nested_samples[i], self.names)
            logpr       = self.log_prior_fn(this_params)
            logl        = np.float(self.results.logl[i])
        
            ns.append(np.append(self.nested_samples[i], [logl, logpr]))
            wt.append(np.float(self.results.logwt[i]-self.results.logz[-1]))
        
        ns      = np.array(ns)
        wt      = np.exp(np.array(wt))
        names   = np.append(self.names , ['logL', 'logPrior'])
        
        # resample nested samples into posterior samples
        self.posterior_samples  = resample(ns, wt)
        self.real_nout          = len(self.posterior_samples)
        
        # extract evidence
        self.logZ       = np.array(self.results.logz)
        self.logZerr    = self.results.logzerr

        logger.info(" - number of posterior samples : {}".format(self.real_nout))

        post_file = open(self.outdir + '/posterior.dat', 'w')

        post_file.write('#')
        for n in range(self.ndim+2):
            post_file.write('{}\t'.format(names[n]))
        post_file.write('\n')

        for i in range(self.real_nout):
            for j in range(self.ndim+2):
                post_file.write('{}\t'.format(self.posterior_samples[i][j]))
            post_file.write('\n')

        post_file.close()
        
        evidence_file = open(self.outdir + '/evidence.dat', 'w')
        evidence_file.write('#\tlogX\tlogZ\tlogZerr\n')
        for xi,zi,ei in zip(self.results.logvol,self.logZ,self.logZerr):
            evidence_file.write('{}\t{}\t{}\n'.format(xi,zi,ei))

        evidence_file.close()
        
    def make_plots(self):
        
        try:
        
            from dynesty import plotting as dyplot
            import matplotlib.pyplot as plt
            
            from ...pipe import ensure_dir
            import os
            auxdir = os.path.join(self.outdir,'dynesty')
            ensure_dir(auxdir)
            
            fig, axes = dyplot.runplot(self.results)
            plt.savefig(auxdir+'/runplot.png', dpi=200)
        
            fig, axes = dyplot.traceplot(self.results, show_titles=True, labels=self.names)
            plt.savefig(auxdir+'/traceplot.png', dpi=200)
        
            fig, axes = dyplot.cornerplot(self.results, color='royalblue', show_titles=True, labels=self.names)
            plt.savefig(auxdir+'/cornerplot.png', dpi=200)

        except ImportError:
            logger.warning("Unable to produce standard plots, check if matplotlib is installed.")

class SamplerDynestyDynamic(SamplerDynesty):
    
    def __initialize__(self, posterior, nbatch=512, **kwargs):

        # initialize dynamic nested parameters
        self.nbatch      = nbatch
        self.init_flag   = False

        # initialize dynesty inference
        super(SamplerDynestyDynamic, self).__initialize__(posterior, **kwargs)
    
        # extract prior samples, ensuring finite logL
        logger.info("Extracting prior samples ...")
        self.p0 = get_prior_samples_dynesty(kwargs['nlive'], self.ndim, self.sampler.loglikelihood, self.sampler.prior_transform)
    
        # set customized proposal
        dynesty.dynesty._SAMPLING["rwalk"] = self.sampler.evolve_point
        dynesty.nestedsamplers._SAMPLING["rwalk"] = self.sampler.evolve_point
        
    def _initialize_sampler(self, like_fn, ptform_fn, kwargs):
        logger.info("Initializing nested sampler ...")
        return dynesty.DynamicNestedSampler(like_fn, ptform_fn, **kwargs)
    
    def __getstate__(self):
        state   = self.__dict__.copy()
        state['sampler'].pool   = None
        state['sampler'].M      = None
        state['sampler'].rstate = None
        return state

    def __run__(self):
        
        # perform initial sampling if necessary
        if not self.init_flag:
            self._run_nested_initial()
        
        logger.info("Completing initial sampling ...")
        logger.info("Running batches with {} live points ...".format(self.nbatch))
        self._run_batches()

        # final store inference
        self.store()
            
    def _store_current_live_points(self):
        self.p0 = [self.sampler.sampler.live_u,
                   self.sampler.sampler.live_v,
                   self.sampler.sampler.live_logl]

    def _run_nested_initial(self):
        
        # check if a non-empty set of live points already exist
        if len(self.p0) == 0:
            # if none, the sampler has to be initialized
            # and the initial samples are extracted from the prior
            live_points = None
        else:
            # otherwise restore previous live points
            # and avoid default reset
            live_points = self.p0
            self.sampler.reset = void

        # Baseline run.
        for results in self.sampler.sample_initial(nlive=self.nlive, dlogz=self.tol, live_points=live_points):
            if self.sampler.it%self.ncheckpoint==0:
                self._store_current_live_points()
                self._last_iter = results
                self.update()

        # Store inference at the end of initial sampling
        self.init_flag = True
        self._last_iter = results
        self.update()

    def __update_batches__(self):

        (worst, ustar, vstar, loglstar, nc, worst_it, boundidx, bounditer, eff) = results

        args = {'it' :      self.sampler.it,
                'eff' :     '{:.2f}'.format(eff),
                'nc' :      nc,
                #'logL' :   '{:.3f}'.format(loglstar),
                'logLmax' : '{:.3f}'.format(np.max(self.sampler.live_logl))}
        return args

    def _run_batches(self):
        
        # use new updating
        self.__update__ = self.__update_batches__
        
        logger.info("Adding batches to the sampling ...")
        from dynesty.dynamicsampler import stopping_function, weight_function

        # Add batches until we hit the stopping criterion.
        while True:
            stop = stopping_function(self.sampler.results)  # evaluate stop
            if not stop:
                logl_bounds = weight_function(self.sampler.results)  # derive bounds
                for results in self.sampler.sample_batch(nlive_new=self.nbatch, logl_bounds=logl_bounds):
                    if self.sampler.it%self.ncheckpoint==0:
                        self._last_iter = results
                        self.update()

                self.sampler.combine_runs()  # add new samples to previous results
            else:
                break

class BajesDynestyProposal(object):

    def __init__(self, maxmcmc, walks=100, nact=5.):

        self.maxmcmc    = maxmcmc
        self.walks      = walks     # minimum number of steps
        self.nact       = nact      # Number of ACT (safety param)

#    def update_rwalk(self, blob):
#        scale = blob['scale']
#        accept, reject = blob['accept'], blob['reject']
#        facc = (1. * accept) / (accept + reject)
#        norm = max(self.facc, 1. - self.facc) * self.ndim
#        scale *= np.exp((facc - self.facc) / norm)
#        return min(scale, np.sqrt(self.ndim))

    def propose(self, args):
        """
            Inspired from bilby-implemented version of dynesty.sampling.sample_rwalk
            The difference is that if chain hits maxmcmc, slice sampling is proposed
        """

        # Unzipping.
        (u, loglstar, axes, scale, prior_transform, loglikelihood, kwargs) = args
        rstate = np.random

        # Bounds
        nonbounded  = kwargs.get('nonbounded', None)
        periodic    = kwargs.get('periodic', None)
        reflective  = kwargs.get('reflective', None)
        old_act     = np.copy(self.walks)

        # Setup.
        n = len(u)
        nc = 0

        # Initialize internal variables
        accept  = 0
        reject  = 0
        nfail   = 0
        act     = np.inf
        u_list  = []
        v_list  = []
        logl_list = []

        ii = 0

        # while .5 * self.nact * act > len(u_list):
        while ii < self.nact * act:

            ii += 1

            # Propose a direction on the unit n-sphere.
            drhat = rstate.randn(n)
            drhat /= np.linalg.norm(drhat)

            # Scale based on dimensionality.
            dr = drhat * rstate.rand() ** (1.0 / n)

            # Transform to proposal distribution.
            du = np.dot(axes, dr)
            u_prop = u + scale * du

            # Wrap periodic parameters
            if periodic is not None:
                u_prop[periodic] = np.mod(u_prop[periodic], 1)
            # Reflect
            if reflective is not None:
                u_prop[reflective] = reflect(u_prop[reflective])

            # Check unit cube constraints.
            if unitcheck(u_prop, nonbounded):
                pass
            else:
                nfail += 1
                # Only start appending to the chain once a single jump is made
                if accept > 0:
                    u_list.append(u_list[-1])
                    v_list.append(v_list[-1])
                    logl_list.append(logl_list[-1])
                continue

            # Check proposed point.
            v_prop      = prior_transform(np.array(u_prop))
            logl_prop   = loglikelihood(np.array(v_prop))

            if logl_prop > loglstar:
                u = u_prop
                v = v_prop
                logl = logl_prop
                accept += 1
                u_list.append(u)
                v_list.append(v)
                logl_list.append(logl)
            else:
                reject += 1
                # Only start appending to the chain once a single jump is made
                if accept > 0:
                    u_list.append(u_list[-1])
                    v_list.append(v_list[-1])
                    logl_list.append(logl_list[-1])

            # If we've taken the minimum number of steps, calculate the ACT
            if accept + reject > self.walks and accept > 1:
                act = estimate_nmcmc(accept_ratio=accept / (accept + reject + nfail), old_act=old_act, maxmcmc=self.maxmcmc)

            # If we've taken too many likelihood evaluations then break
            if accept + reject > self.maxmcmc:
                logger.warning("Hit maxmcmc iterations ({}) with accept={}, reject={}, and nfail={}.".format(self.maxmcmc, accept, reject, nfail))
                break

        # If the act is finite, pick randomly from within the chain
        if np.isfinite(act) and int(.5 * self.nact * act) < len(u_list):
            idx = np.random.randint(int(.5 * self.nact * act), len(u_list))
            u = u_list[idx]
            v = v_list[idx]
            logl = logl_list[idx]
        else:
            logger.debug("Unable to find a new point using random walk. Performing slice step.")
            u, v, logl, _nc, blb = self.sample_rslice(args)
            nc += _nc

        blob    = {'accept': accept, 'reject': reject, 'fail': nfail, 'scale': scale}
        ncall   = accept + reject + nc

        return u, v, logl, ncall, blob

    def sample_rslice(self, args):

        # Unzipping.
        (u, loglstar, axes, scale, prior_transform, loglikelihood, kwargs) = args
        rstate = np.random

        # Periodicity.
        nonperiodic = kwargs.get('reflective', None)

        # Setup.
        n = len(u)
        slices = len(u)//2
        nc = 0
        nexpand = 0
        ncontract = 0
        fscale = []

        # Modifying axes and computing lengths.
        axes = scale * axes.T  # scale based on past tuning
        axlens = [np.linalg.norm(axis) for axis in axes]

        # Slice sampling loop.
        for it in range(slices):

            # Propose a direction on the unit n-sphere.
            drhat = rstate.randn(n)
            drhat /= np.linalg.norm(drhat)

            # Transform and scale based on past tuning.
            axis = np.dot(axes, drhat) * scale
            axlen = np.linalg.norm(axis)

            # Define starting "window".
            r = rstate.rand()  # initial scale/offset

            u_l = u - r * axis  # left bound
            if unitcheck(u_l, nonperiodic):
                v_l = prior_transform(np.array(u_l))
                logl_l = loglikelihood(np.array(v_l))
            else:
                logl_l = -np.inf
            nc += 1
            nexpand += 1

            u_r = u + (1 - r) * axis  # right bound
            if unitcheck(u_r, nonperiodic):
                v_r = prior_transform(np.array(u_r))
                logl_r = loglikelihood(np.array(v_r))
            else:
                logl_r = -np.inf
            nc += 1
            nexpand += 1


            # "Stepping out" the left and right bounds.
            while logl_l > loglstar:
                u_l -= axis
                if unitcheck(u_l, nonperiodic):
                    v_l = prior_transform(np.array(u_l))
                    logl_l = loglikelihood(np.array(v_l))
                else:
                    logl_l = -np.inf
                nc += 1
                nexpand += 1
            while logl_r > loglstar:
                u_r += axis
                if unitcheck(u_r, nonperiodic):
                    v_r = prior_transform(np.array(u_r))
                    logl_r = loglikelihood(np.array(v_r))
                else:
                    logl_r = -np.inf
                nc += 1
                nexpand += 1

            # Sample within limits. If the sample is not valid, shrink
            # the limits until we hit the `loglstar` bound.
            window_init = np.linalg.norm(u_r - u_l)  # initial window size

            while True:
                # Define slice and window.
                u_hat = u_r - u_l
                window = np.linalg.norm(u_hat)

                # Check if the slice has shrunk to be ridiculously small.
                if window < 1e-10 * window_init:
                    logger.warning("Slice sampling appears to be stuck. Returning uniform sample.")
                    u = np.random.uniform(size=n)
                    v = prior_transform(u)
                    logl = loglikelihood(v)
                    blob = {'fscale': 1., 'nexpand': nexpand, 'ncontract': ncontract}
                    return u, v, logl, nc, blob

                # Propose new position.
                u_prop = u_l + rstate.rand() * u_hat  # scale from left
                if unitcheck(u_prop, nonperiodic):
                    v_prop = prior_transform(np.array(u_prop))
                    logl_prop = loglikelihood(np.array(v_prop))
                else:
                    logl_prop = -np.inf
                nc += 1
                ncontract += 1

                # If we succeed, move to the new position.
                if logl_prop > loglstar:
                    fscale.append(window / axlen)
                    u = u_prop
                    break
                # If we fail, check if the new point is to the left/right of
                # our original point along our proposal axis and update
                # the bounds accordingly.
                else:
                    s = np.dot(u_prop - u, u_hat)  # check sign (+/-)
                    if s < 0:  # left
                        u_l = u_prop
                    elif s > 0:  # right
                        u_r = u_prop
                    else:
                        # If `s = 0` something has gone horribly wrong.
                        logger.warning("Slice sampler has failed to find a valid point. Returning uniform sample.")
                        u = np.random.uniform(size=n)
                        v = prior_transform(u)
                        logl = loglikelihood(v)
                        blob = {'fscale': 1., 'nexpand': nexpand, 'ncontract': ncontract}
                        return u, v, logl, nc, blob

        blob = {'fscale': np.mean(fscale), 'nexpand': nexpand, 'ncontract': ncontract}
        return u_prop, v_prop, logl_prop, nc, blob

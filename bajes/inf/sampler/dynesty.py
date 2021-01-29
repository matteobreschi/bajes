from __future__ import division, unicode_literals, absolute_import
import numpy as np
import signal
import tracemalloc
import os

import logging
logger = logging.getLogger(__name__)

import dynesty

from dynesty.utils import unitcheck

from ..utils import estimate_nmcmc, list_2_dict
from ...pipe import data_container, display_memory_usage


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

class SamplerNest(object):
    
    def __init__(self, posterior,
                 nlive, tolerance=0.1, ncheckpoint=0,
                 # bounding
                 bound_method='multi', vol_check=8., vol_dec=0.5,
                 # update
                 bootstrap=0, enlarge=1.5, facc=0.5, update_interval=None,
                 # proposal
                 proposals=None, nact = 5., maxmcmc=4096, minmcmc=32,
                 # first update
                 first_min_ncall = None, first_min_eff = 10,
                 # others
                 nprocs=None, pool=None, use_slice=False, use_gw=False,
                 outdir='./', resume='/resume.pkl', seed=None, **kwargs):
        
        self.resume = resume
        self.outdir = outdir

        # restore inference from existing container
        if os.path.exists(self.outdir + self.resume):
            self.restore_inference(pool)
    
        # initialize a new inference
        else:
            
            # initialize signal
            try:
                signal.signal(signal.SIGTERM,   self.store_inference_and_exit)
                signal.signal(signal.SIGINT,    self.store_inference_and_exit)
                signal.signal(signal.SIGALRM,   self.store_inference_and_exit)
            except AttributeError:
                logger.warning("Impossible to set signal attributes.")

            # initialize nested parameters
            self.nlive          = nlive
            self.tol            = tolerance
            
            # auxiliary arguments
            self.names = posterior.prior.names
            self.ndim  = len(self.names)
            self.log_prior_fn = posterior.log_prior
            
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
            
            # initialize seed
            if seed == None:
                import time
                self.seed = int(time.time())
            else:
                self.seed = seed
            np.random.seed(self.seed)
            
            if self.nlive < self.ndim*(self.ndim-1)//2:
                logger.warning("Given number of live points < Ndim*(Ndim-1)/2. This may generate problems in the exploration of the parameters space.")

            # set up periodic and reflective boundaries
            periodic_inds   = np.concatenate(np.where(np.array(posterior.prior.periodics) == 1))
            reflective_inds = np.concatenate(np.where(np.array(posterior.prior.periodics) == 0))

            # initialize proposals
            if proposals == None:
                logger.info("Initializing proposal methods ...")
                proposals = initialize_proposals(maxmcmc, minmcmc, nact)

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
            self.sampler    = self.initialize_sampler(like_fn, ptform_fn, sampler_kwargs)

            # clean up sampler
            del self.sampler.cite
            del self.sampler.kwargs['cite']
            self.sampler.rstate = np.random

            # set proposal
            self.sampler.evolve_point = proposals.propose

    def __getstate__(self):
        self_dict = self.__dict__.copy()
#        if 'sampler' in list(self_dict.keys()):
#            self_dict['sampler'].pool   = None
#            self_dict['sampler'].M      = None
#            self_dict['sampler'].rstate = None
        return self_dict

    def initialize_sampler(self, like_fn, ptform_fn, kwargs):
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
            
    def store_inference_and_exit(self, signum=None, frame=None):
        # exit function when signal is revealed
        logger.info("Run interrupted by signal {}, checkpoint and exit.".format(signum))
        os._exit(signum)
            
    def restore_inference(self, pool):

        # extract container
        logger.info("Restoring inference from existing container ...")
        dc                  = data_container(self.outdir + self.resume)
        container           = dc.load()
        
        # sampler check
        if container.tag != 'nest':
            logger.error("Container carries a {} inference, while NEST was requested.".format(container.tag.upper()))
            raise AttributeError("Container carries a {} inference, while NEST was requested.".format(container.tag.upper()))

        previous_inference  = container.inference

        # extract previous variables and methods
        for kw in list(previous_inference.__dict__.keys()):
            self.__setattr__(kw, previous_inference.__dict__[kw])

        # re-initialize pool
        self.sampler.pool   = pool
        self.sampler.M      = pool.map

        # re-initialize seed
        if self.seed == None:
            import time
            self.seed = int(time.time())
        np.random.seed(self.seed)
        self.sampler.rstate = np.random

        # re-initialize signal
        try:
            signal.signal(signal.SIGTERM,   self.store_inference_and_exit)
            signal.signal(signal.SIGINT,    self.store_inference_and_exit)
            signal.signal(signal.SIGALRM,   self.store_inference_and_exit)
        except AttributeError:
            logger.warning("Impossible to set signal attributes.")
                
    def store_inference(self):
        # save inference in pickle file
        dc = data_container(self.outdir+self.resume)
        dc.store('tag', 'nest')
        dc.store('inference', self)
        dc.save()

    def run(self):
        
        # run the sampler
        logger.info("Running {} live points ...".format(self.nlive))
        
        for results in self.sampler.sample(dlogz=self.tol,save_samples=True,add_live=False):
            
            if self.sampler.it%self.ncheckpoint==0:
                
                (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results
                self.update_sampler([eff/100.,nc,loglstar,logz,h,delta_logz])
    
                if tracemalloc.is_tracing():
                    display_memory_usage(tracemalloc.take_snapshot())
                    tracemalloc.clear_traces()

        # add live points to nested samples
        logger.info("Adding live points in nested samples")
        self.sampler.add_final_live(print_progress=False)

        # final store inference
        self.store_inference()

    def update_sampler(self, args):

        acc, nc, logl, logz, h, d_logz = args

        # store inference
        if self.store_flag:
            self.store_inference()

        logger.info(" - it : {:d} - eff : {:.3f} - ncall : {:.0f} - logL : {:.3g} - logLmax : {:.3g} - logZ : {:.3g} - H : {:.3g} - dlogZ : {:.3g}".format(self.sampler.it,acc,nc,logl,np.max(self.sampler.live_logl),logz,h,d_logz))

    def get_posterior(self):
        
        self.results            = self.sampler.results
        self.nested_samples     = self.results.samples
        logger.info(" - number of nested samples : {}".format(len(self.nested_samples)))
        
        # extract posteriors
        ns = []
        wt = []
        scale = np.array(self.sampler.saved_scale)
        for i in range(len(self.nested_samples)):
            
            # start appending from the first update
            # if scale[i] < 1. :
            
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
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning("Impossible to produce standard plots. Cannot import matplotlib.")
                
        try:

            fig = plt.figure()
            plt.plot(self.results.logvol, self.results.logl)
            plt.xlim(( np.min(self.results.logvol),np.max(self.results.logvol) ))
            plt.ylim((0.,1.1*(np.max(self.results.logl))))
            plt.ylabel('lnL - lnZnoise')
            plt.xlabel('lnX')
            plt.savefig(self.outdir+'/lnL_lnX.png', dpi=200)

            plt.close()
        
            fig = plt.figure()
            plt.fill_between(self.results.logvol, self.logBF-self.logZerr, self.logBF+self.logZerr, alpha=0.6, color='royalblue')
            plt.plot(self.results.logvol, self.logBF, color='navy', label='logBF')
            plt.plot(self.results.logvol, self.results.logl, color='slateblue', label='logL', ls='--')
            plt.xlim(( np.min(self.results.logvol),np.max(self.results.logvol) ))
            
            ylim_max = np.max([ np.max(self.logBF) ,np.max(self.results.logl)])
            plt.ylim((0.,1.2*ylim_max))
            plt.xlabel('logX')
            plt.savefig(self.outdir+'/lnBF_lnX.png', dpi=200)
            
            plt.close()

        except Exception:
            pass

class SamplerDyNest(SamplerNest):
    
    def __init__(self, posterior, nbatch=512, **kwargs):

        # initialize dynamic nested parameters
        self.nbatch      = nbatch
        self.init_flag   = False

        # initialize dynesty inference
        super(SamplerDyNest, self).__init__(posterior, **kwargs)
    
        # extract prior samples, ensuring finite logL
        logger.info("Extracting prior samples ...")
        self.p0 = get_prior_samples_dynesty(kwargs['nlive'], self.ndim, self.sampler.loglikelihood, self.sampler.prior_transform)
    
        # set customized proposal
        dynesty.dynesty._SAMPLING["rwalk"] = self.sampler.evolve_point
        dynesty.nestedsamplers._SAMPLING["rwalk"] = self.sampler.evolve_point
    
    def __getstate__(self):
        # get __dict__ of parent class
        inher_dict  = SamplerNest.__getstate__(self)
        # get __dict__ of this class
        self_dict   = self.__dict__.copy()
        # merge them
        full_dict   = {**inher_dict, **self_dict}

#        if 'sampler' in list(full_dict.keys()):
#            full_dict['sampler'].pool   = None
#            full_dict['sampler'].M      = None
#            full_dict['sampler'].rstate = None

        return full_dict
        
    def initialize_sampler(self, like_fn, ptform_fn, kwargs):
        logger.info("Initializing nested sampler ...")
        return dynesty.DynamicNestedSampler(like_fn, ptform_fn, **kwargs)
    
    def restore_inference(self, pool):

        # extract container
        logger.info("Restoring inference from existing container ...")
        dc                  = data_container(self.outdir + self.resume)
        container           = dc.load()

        # sampler check
        if container.tag != 'dynest':
            logger.error("Container carries a {} inference, while DYNEST was requested.".format(container.tag.upper()))
            raise AttributeError("Container carries a {} inference, while DYNEST was requested.".format(container.tag.upper()))

        previous_inference  = container.inference

        # extract previous variables and methods
        for kw in list(previous_inference.__dict__.keys()):
            self.__setattr__(kw, previous_inference.__dict__[kw])

        # re-initialize pool
        self.sampler.pool   = pool
        self.sampler.M      = pool.map

        # re-initialize seed
        if self.seed == None:
            import time
            self.seed = int(time.time())
        np.random.seed(self.seed)
        self.sampler.rstate = np.random

        # re-initialize signal
        try:
            signal.signal(signal.SIGTERM,   self.store_inference_and_exit)
            signal.signal(signal.SIGINT,    self.store_inference_and_exit)
            signal.signal(signal.SIGALRM,   self.store_inference_and_exit)
        except AttributeError:
            logger.warning("Impossible to set signal attributes.")

    def store_inference(self):
        # save inference in pickle file
        dc = data_container(self.outdir+self.resume)
        dc.store('tag', 'dynest')
        dc.store('inference', self)
        dc.save()
    
    def update_sampler(self, args):
        
        acc, nc, logl, logz, h, d_logz = args

        # store inference
        if self.store_flag:
            self.store_inference()

        logger.info(" - it : {:d} - eff : {:.3f} - ncall : {:.0f} - logL : {:.3g} - logLmax : {:.3g} - logZ : {:.3g} - H : {:.3g} - dlogZ : {:.3g}".format(self.sampler.it,acc,nc,logl,np.max(self.sampler.live_logl),logz,h,d_logz))

    def run(self):
        
        logger.info("Running {} live points ...".format(self.nlive))
        
        # perform initial sampling if necessary
        if not self.init_flag:
            self.run_nested_initial()
        
        logger.info("Completing initial sampling ...")
        logger.info("Running batches with {} live points ...".format(self.nbatch))
        self.run_batches()

        # final store inference
        self.store_inference()
            
    def store_current_live_points(self):

        # Sorting remaining live points.
        lsort_idx = np.argsort(self.sampler.sampler.live_logl)

        # Add contributions from the remaining live points in order
        # from the lowest to the highest log-likelihoods.
        live_u = [self.sampler.sampler.live_u[i] for i in lsort_idx]
        live_v = [self.sampler.sampler.live_v[i] for i in lsort_idx]
        live_l = [self.sampler.sampler.live_logl[i] for i in lsort_idx]
        self.p0 = [np.array(live_u),np.array(live_v),np.array(live_l)]

    def run_nested_initial(self):
        
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
                self.store_current_live_points()
                (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results
                self.update_sampler([eff/100.,nc,loglstar,logz,h,delta_logz])

                if tracemalloc.is_tracing():
                    display_memory_usage(tracemalloc.take_snapshot())
                    tracemalloc.clear_traces()

        # Store inference at the end of initial sampling
        self.init_flag = True
        (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results
        self.update_sampler([eff/100.,nc,loglstar,logz,h,delta_logz])

    def run_batches(self):
        
        logger.info("Adding batches to the sampling ...")
        from dynesty.dynamicsampler import stopping_function, weight_function

        # Add batches until we hit the stopping criterion.
        while True:
            stop = stopping_function(self.sampler.results)  # evaluate stop
            if not stop:
                logl_bounds = weight_function(self.sampler.results)  # derive bounds
                for results in self.sampler.sample_batch(nlive_new=self.nbatch, logl_bounds=logl_bounds):
                    
                    if self.sampler.it%self.ncheckpoint==0:
                        
                        (worst, ustar, vstar, loglstar, nc, worst_it, boundidx, bounditer, eff) = results
                        self.update_sampler([eff/100.,nc,loglstar,self.sampler.results.logz[-1],0.,self.sampler.results.logzerr[-1]])
            
                        if tracemalloc.is_tracing():
                            display_memory_usage(tracemalloc.take_snapshot())
                            tracemalloc.clear_traces()

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

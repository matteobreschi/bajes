from __future__ import division, unicode_literals, absolute_import
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.special import logsumexp

from ultranest import ReactiveNestedSampler

from . import SamplerBody

def index_from_chain(chain, param):

    res = ((chain - param)**2.).sum(axis=1)
    ind = int(np.argmin(res))

    if res[ind] == 0:
        return ind
    elif res[ind] < 1e-32:
        logger.debug("Index for parameter {} has non-zero residual lower than threshold (1e-32)".format(param))
        return ind
    else:
        logger.warning("Index for parameter {} has non-zero residual.".format(param))
        return ind

class _vectorized_posterior(object):

    def __init__(self, posterior, pool):
        self._post = posterior
        self._pool = pool
        self._mapf = self._pool.map

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_pool'] = None
        state['_mapf'] = map
        return state

    def log_like(self, x):
        return np.array(list(self._mapf(self._post.log_like, x)))

    def prior_transform(self, x):
        return np.array(list(self._mapf(self._post.prior_transform, x)))

class _ReactiveNestedSampler(ReactiveNestedSampler):

    def __getstate__(self):
        state               = self.__dict__.copy()
        state['comm']       = None
        state['pointstore'] = None
        state['transform']  = None
        state['loglike']    = None
        return state

class SamplerUltraNest(SamplerBody):

    def __initialize__(self, posterior,
                       nlive, tolerance=0.1,
                       maxmcmc=4096, minmcmc=128,
                       nout=10000, dkl=0.5,
                       z_frac=None, Lepsilon=0.5,
                       num_bootstraps=30, nlive_cluster=None,
                       proposals_kwargs={'use_slice': False},
                       pool=None, **kwargs):

        self.log_prior_fn = posterior.log_prior

        if nlive < len(posterior.prior.names)*(len(posterior.prior.names)-1)//2:
            logger.warning("Given number of live points < Ndim*(Ndim-1)/2. This may generate problems in the exploration of the parameters space.")

        if nlive_cluster == None:
            nlive_cluster = nlive//10

        # if self.store_flag:
        import os
        log_dir = os.path.join(self.outdir, 'ultranest')

        sampler_kwargs  = { 'log_dir':          log_dir,
                            'vectorized':       False,
                            'ndraw_min':        minmcmc,
                            'ndraw_max' :       maxmcmc,
                            'num_bootstraps':   num_bootstraps,
                            'resume':           'resume',
                            'storage_backend':  'hdf5',
                            'wrapped_params' :  [bool(i) for i in posterior.prior.periodics]}

        # check vectorization
        if pool is not None:

            # check it is actually a multiprocessing pool
            from multiprocessing.pool import Pool
            if not isinstance(pool, Pool):
                logger.error("Unable to set parallel processes, pool argument for UltraNest sampler must be a multiprocessing pool.")
                raise AttributeError("Unable to set parallel processes, pool argument for UltraNest sampler must be a multiprocessing pool.")

            # turn on vectorization
            logger.info("Activating vectorized posterior function...")
            sampler_kwargs['vectorized'] = True

            # wrap vectorized posterior
            posterior = _vectorized_posterior(posterior=posterior, pool=pool)

        # set frac_remain argument
        if z_frac == None:
            z_frac = 1. - np.exp(-tolerance)

        # set update_interval_volume_fraction argument
        update_vol_frac = max(0.,1. - np.exp(-self.ncheckpoint/nlive))

        # save runner arguments
        self.run_kwargs = { 'update_interval_volume_fraction':  update_vol_frac,
                            'viz_callback':                     self.update,
                            'show_status':                      False,
                            'dlogz':                            tolerance,
                            'dKL':                              dkl,
                            'min_ess':                          nout,
                            'frac_remain':                      z_frac,
                            'Lepsilon':                         Lepsilon,
                            'min_num_live_points':              nlive,
                            'cluster_num_live_points':          nlive_cluster}

        logger.info("Initializing nested sampler ...")
        self.sampler = _ReactiveNestedSampler(self.names,
                                              posterior.log_like,
                                              posterior.prior_transform,
                                              **sampler_kwargs)

        logger.info("Initializing proposal method ...")
        import ultranest.stepsampler as stepsampler
        if proposals_kwargs['use_slice']:
            logger.debug("Using slice sampling method")
            self.sampler.stepsampler = stepsampler.RegionBallSliceSampler(nsteps=minmcmc)
        else:
            logger.debug("Using random-walk sampling method")
            self.sampler.stepsampler = stepsampler.RegionMHSampler(nsteps=minmcmc)

        # set unique logger
        self.sampler.logger = logger

        # avoid mpi.comm if mpi is not used
        if not self.sampler.use_mpi:
            self.sampler.comm = None

    def __restore__(self, pool, **kwargs):

        # re-initialize vectorized probabilities
        if pool is not None:

            if not self.sampler.draw_multiple:
                logger.warning("Inconsitency revealed between current settings and resumed object, turning on multiple draws.")
                self.sampler.draw_multiple = True

            posterior   = _vectorized_posterior(posterior=kwargs['posterior'], pool=pool)

        # re-initialize serial probabilities
        else:
            if self.sampler.draw_multiple:
                logger.warning("Inconsitency revealed between current settings and resumed object, turning off multiple draws.")
                self.sampler.draw_multiple = False

            posterior   = kwargs['posterior']

        # re-initialize sampler
        wrapped_params = np.zeros(self.ndim)
        wrapped_params[self.sampler.wrapped_axes] = 1.
        sampler_kwargs  = { 'log_dir':          self.outdir+'/ultranest',
                            'vectorized':       self.sampler.draw_multiple,
                            'ndraw_min':        self.sampler.ndraw_min,
                            'ndraw_max' :       self.sampler.ndraw_max,
                            'num_bootstraps':   self.sampler.num_bootstraps,
                            'resume':           'resume',
                            'storage_backend':  'hdf5',
                            'wrapped_params' :  wrapped_params}

        # check proposal method
        import ultranest.stepsampler as stepsampler
        use_slice = False
        if isinstance(self.sampler.stepsampler, stepsampler.RegionBallSliceSampler):
            use_slice = True

        logger.debug("Re-initializing sampler ...")
        self.sampler = _ReactiveNestedSampler(self.names,
                                              posterior.log_like,
                                              posterior.prior_transform,
                                              **sampler_kwargs)

        if use_slice:
            logger.debug("Using slice sampling method")
            self.sampler.stepsampler = stepsampler.RegionBallSliceSampler(nsteps=self.sampler.ndraw_min)
        else:
            logger.debug("Using random-walk sampling method")
            self.sampler.stepsampler = stepsampler.RegionMHSampler(nsteps=self.sampler.ndraw_min)

        # set unique logger
        self.sampler.logger = logger

        # avoid mpi.comm if mpi is not used
        if not self.sampler.use_mpi:
            self.sampler.comm = None

    def __update__(self, points, info, **others):

        #zf      = 1.0 / (1 + np.exp(info['logz'] - info['logz_remain']))
        dz      = np.logaddexp(info['logz'], info['logz_remain']) - info['logz']
        args    = { 'it' :      info['it'],
                    'eff' :     '{:.2f}%'.format(100.*info['it']/self.sampler.ncall),
                    'acl' :     info['order_test_correlation'],
                    'logV' :    '{:.2f}'.format(info['logvol']),
                    'logLmax' : '{:.3f}'.format(np.max(points['logl'])),
                    'logZ' :    '{:.3f}'.format(info['logz']),
                    'dZ' :      '{:.3g}'.format(dz)}

        return args

    def __run__(self):

        # final store inference
        for result in self.sampler.run_iter(**self.run_kwargs):
            self.results = result

        # final store inference
        self.store()

    def get_posterior(self):

        logger.info('  H = {:.4f} +- {:.4f}'.format(self.results['H'],self.results['Herr']))

        if self.results['insertion_order_MWW_test']['converged'] and np.isfinite(self.results['insertion_order_MWW_test']['independent_iterations']):
            logger.info('Converged chain with correlation %(independent_iterations)s' % (self.results['insertion_order_MWW_test']))
        else:
            if not self.results['insertion_order_MWW_test']['converged']:
                logger.warning('Sampler did not converged, consider to increase the number of mcmc iterations.')
            if not np.isfinite(self.results['insertion_order_MWW_test']['independent_iterations']):
                logger.warning('Samples are highly correlated, consider to increase the number of mcmc iterations.')

        # get posterior samples
        names                   = np.append(self.names , ['logL', 'logPrior'])
        samples                 = np.array(self.results['samples'])
        self.posterior_samples  = []
        weight_samples          = self.results['weighted_samples']['points']
        for pi in samples:
            logpr   = self.log_prior_fn(pi)
            logl    = self.results['weighted_samples']['logl'][index_from_chain(weight_samples,pi)]
            self.posterior_samples.append(np.append(pi, [logl, logpr]))
        self.real_nout  = len(self.posterior_samples)

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
        evidence_file.write('Final logZ = {} +- {}\n'.format(self.results['logz'],self.results['logzerr']))
        evidence_file.write(' * single logZ = {} +- {}\n'.format(self.results['logz_single'],self.results['logzerr_single']))
        evidence_file.write(' * b.s.   logZ = {} +- {}\n'.format(self.results['logz_bs'],self.results['logzerr_bs']))
        evidence_file.write(' * tail   logZ +- {}\n'.format(self.results['logzerr_tail']))
        evidence_file.write(' * information = {} +- {}\n'.format(self.results['H'],self.results['Herr']))
        evidence_file.close()

    def make_plots(self):

        try:

            from ultranest.plot import cornerplot
            cornerplot(self.results)
            self.sampler.plot_trace()

        except Exception:
            logger.info("Standard plots of ultranest are not procuded if checkpoint is disabled.")
            logger.warning("Unable to produce standard plots, check if matplotlib and corner are installed.")

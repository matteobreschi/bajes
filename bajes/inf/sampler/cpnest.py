from __future__ import division, unicode_literals, absolute_import
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

import cpnest
import cpnest.model

from cpnest.parameter import LivePoint
from cpnest.proposal import EnsembleProposal, ProposalCycle

from .proposal import ModelTuple, _init_proposal_methods

from ..utils import dict_2_list
from ...pipe import data_container

def store_inference(path):
    # save inference in pickle file
    dc = data_container(path)
    dc.store('tag', 'cpnest')
    dc.save()

def check_container(path):
    # extract information from container
    dc          = data_container(path)
    container   = dc.load()

    # sampler check
    if container.tag != 'cpnest':
        logger.error("Container carries a {} inference, while CPNEST was requested.".format(container.tag.upper()))
        raise AttributeError("Container carries a {} inference, while CPNEST was requested.".format(container.tag.upper()))

def initialize_proposals(post, use_slice=False, use_gw=False):
    props = {'eig': 25., 'dif': 25., 'str': 20., 'wlk': 15., 'kde': 10., 'pri': 5.}
    prop_kwargs  = {}
    
    if use_slice:
        props['slc'] = 30.
    
    if use_gw:
        props['gwt'] = 20.
        prop_kwargs['like'] = post.like
        prop_kwargs['dets'] = post.like.dets
    return BajesCPNestProposal(post, props, **prop_kwargs)

class CPNestProposalWrapper(EnsembleProposal):
    """
        Wrapper for standard proposal function into cpnest-oriented proposal function
    """
    def __init__(self, proposal, model, frac=4):
        self._proposal  = proposal
        self._model     = model
        self._frac      = int(frac)
        super(CPNestProposalWrapper,self).__init__()

    def get_sample(self,old):
        """
            Parameters
            ----------
            old : :obj:`cpnest.parameter.LivePoint`

            Returns
            ----------
            out: :obj:`cpnest.parameter.LivePoint`
            """
        Ne = len(self.ensemble)
        ind = np.random.choice(np.arange(Ne),size=Ne//self._frac)
        s = np.array([[old[ni] for ni in old.names]])
        c = np.array([list(map(lambda i: [self.ensemble[i][n] for n in old.names], ind))])
        p = np.array([old.logL])

        q, f = self._proposal.get_proposal(s, c, p, self._model)
        self.log_J = f[0]
        return LivePoint(names=old.names, d=np.concatenate(q))

class BajesCPNestProposal(ProposalCycle):
    """
    A default proposal cycle for GW inference
    """

    def __init__(self, post, props=None, **kwargs):
        """
            Proposal cycle object for MCMC sampler
            Arguments:
            - post    : posterior object
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

        self.log_like = post.log_like
        model = ModelTuple(map_fn=map, compute_log_prob_fn=self._compute, random=np.random)
        _proposals, weights = _init_proposal_methods(post.prior, props=props, **kwargs)
        proposals = [CPNestProposalWrapper(proposal = pi, model=model) for pi in _proposals]

        super(BajesCPNestProposal,self).__init__(proposals, weights)

    def _compute(self, s):
        lnl = list(map(self.log_like, (s[i] for i in range(len(s)))))
        return np.array(lnl), None
        
    def __call__(self):
        return self

class SamplerCPNest(cpnest.CPNest):
    
    #def __init__(self, model, **kwargs):
    #super(BajesCPNest,self).__init__(model, **kwargs)

    def get_posterior(self):
        
        self.get_posterior_samples(filename=os.path.abspath(self.output+'/posterior.dat'))

        logZ        = self.NS.logZ
        infogain    = self.NS.state.info
        logZerr     = np.sqrt(infogain/self.nlive)
        
        from ...pipe import execute_bash
        execute_bash('mv {} {}'.format(os.path.abspath(self.output+'/posterior.dat'),
                                       os.path.abspath(self.output+'/../posterior.dat')))

        evidence_file = open(os.path.abspath(self.output+'/../evidence.dat'), 'w')
        evidence_file.write('logZ   = {} +- {}\n'.format(logZ,logZerr))
        evidence_file.close()

    def make_plots(self):
        self.plot()

class CPNestModel(cpnest.model.Model):
    """
        Wrapper model object for cpnest
    """

    def __init__(self, posterior):

        self.like   = posterior.like
        self.prior  = posterior.prior
        self.names  = self.prior.names
        self.bounds = self.prior.bounds

    def log_likelihood(self,x):
        p = self.prior.this_sample({n : x[n] for n in self.prior.names})
        return self.like.log_like(p)

    def log_prior(self,x):
        if self.prior.in_bounds(x):
            return self.prior.log_prior(x)
        else:
            return -np.inf

def _WrapSamplerCPNest(engine, posterior, nlive, tolerance=0.1, maxmcmc=4096, poolsize=None,
                       proposals=None, proposals_kwargs={'use_slice': False, 'use_gw': False},
                       nprocs=1,  ncheckpoint=None,
                       outdir='./', resume='/resume.pkl', seed=None, **kwargs):

        # set cpnest output
        from ...pipe import ensure_dir
        cpnest_outdir = os.path.join(outdir, 'cpnest')
        ensure_dir(cpnest_outdir)
            
        if nlive < len(posterior.prior.names)*(len(posterior.prior.names)-1)//2:
            logger.warning("Given number of live points < Ndim*(Ndim-1)/2. This may generate problems in the exploration of the parameters space.")
        
        # check inference stored in the container
        if os.path.exists(outdir+resume):
            check_container(outdir+resume)
        else:
            # generate a fictictious pickle,
            # containing only the tag (for a possible wrong resuming)
            store_inference(outdir+resume)

        # initialize proposals
        if proposals == None:
            logger.info("Initializing proposal methods ...")
            proposals = initialize_proposals(posterior, **proposals_kwargs)


        # initialize wrapper model for cpnest
        logger.info("Initializing model ...")
        model = CPNestModel(posterior)

        # initialize sampler
        if ncheckpoint == 0:
            ncheckpoint = None

        sampler =  SamplerCPNest(model,resume=True,proposals=dict(mhs=proposals),verbose=2,
                                 nlive=nlive,maxmcmc=maxmcmc,poolsize=poolsize,nthreads=nprocs,
                                 n_periodic_checkpoint=ncheckpoint,output=cpnest_outdir,seed=seed)

        # set arguments
        sampler.NS.tolerance    = tolerance
        sampler.logger          = logger
        sampler.NS.logger       = logger
        return sampler

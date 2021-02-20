#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import logging
logger = logging.getLogger(__name__)

import os
import signal
import tracemalloc
import numpy as np

from ...pipe import data_container, display_memory_usage

class SamplerBody(object):

    def __init__(self, engine, posterior,
                 ncheckpoint=0,
                 outdir='./',
                 resume='/resume.pkl',
                 seed=None,
                 rank = 0,
                 **kwargs):

        self._engine = engine
        self.resume = resume
        self.outdir = outdir

        # restore inference from existing container
        if os.path.exists(self.outdir + self.resume) and (rank==0):
            
            kwargs['posterior'] = self._restore_posterior()
            self.restore(**kwargs)

        # initialize a new inference
        else:

            # store posterior object
            logger.debug("Initializing sampler on rank {}".format(rank))
            if rank == 0:
                self._first_store(posterior)

            # initialize signal handler
            try:
                signal.signal(signal.SIGTERM, self.store_and_exit)
                signal.signal(signal.SIGINT, self.store_and_exit)
                signal.signal(signal.SIGALRM, self.store_and_exit)
            except AttributeError:
                logger.warning("Unable to set signal attributes.")

            # auxiliary variables
            self.names  = posterior.prior.names
            self.bounds = posterior.prior.bounds
            self.ndim   = len(self.names)

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

            # initialize random seed
            if seed == None:
                import time
                self.seed = int(time.time())
            else:
                self.seed = seed
            np.random.seed(self.seed)

            #initialize specific sampler
            self.__initialize__(posterior, **kwargs)

    def __getstate__(self):
        return self.__dict__.copy()

    def __initialize__(self, posterior, **kwargs):
        # initialize specific sampler
        pass
    
    def __restore__(self, **kwargs):
        # restore specific sampler
        pass
    
    def __run__(self):
        # run specific sampler
        pass
    
    def __update__(self):
        # update specific sampler
        return {}

    def restore(self, **kwargs):

        # extract container
        logger.info("Restoring inference from existing container ...")
        dc          = data_container(self.outdir + self.resume)
        container   = dc.load()
        
        # sampler check
        if container.tag != self._engine:
            logger.error("Container carries a {} inference, while {} was requested.".format(container.tag.upper(), self._engine.upper()))
            raise AttributeError("Container carries a {} inference, while {} was requested.".format(container.tag.upper(), self._engine.upper()))

        # re-initialize signal
        try:
            signal.signal(signal.SIGTERM,   self.store_and_exit)
            signal.signal(signal.SIGINT,    self.store_and_exit)
            signal.signal(signal.SIGALRM,   self.store_and_exit)
        except AttributeError:
            logger.warning("Unable to set signal attributes.")

        # extract previous variables and methods
        previous_inference  = container.inference
        for kw in list(previous_inference.keys()):
            logger.debug("Setting {} attribute ...".format(kw))
            self.__setattr__(kw, previous_inference[kw])

        # re-initialize seed
        np.random.seed(self.seed)
            
        # restore specific sampler
        self.__restore__(**kwargs)

    def store_and_exit(self, signum=None, frame=None):
        # exit function when signal is revealed
        logger.warning("Run interrupted by signal {}, checkpoint and exit.".format(signum))
        try:
            self.store()
        except Exception:
            pass
        os._exit(signum)

    def store(self):
        # save inference in pickle file
        logger.debug("Storing sampler in pickle ...")
        dc = data_container(self.outdir+self.resume)
        state = self.__getstate__()
        logger.debug("Storing following arguments: {}".format(', '.join(list(state.keys()))))
        dc.store('inference', state)
        dc.save()
    
    def _first_store(self, posterior):
        # save inference in pickle file
        logger.debug("Storing posterior ({}) in pickle ...".format(posterior))
        dc = data_container(self.outdir+self.resume)
        dc.store('tag', self._engine)
        dc.store('posterior', posterior)
        dc.save()
    
    def _restore_posterior(self):
        # save inference in pickle file
        logger.debug("Extracting posterior from existing file ...")
        dc          = data_container(self.outdir+self.resume)
        container   = dc.load()
        return container.posterior

    def run(self):
        # run the chains
        logger.info("Running {} sampler ...".format(self._engine))
        self.__run__()

    def update(self, **kwargs):
        
        # get arguments
        prints = self.__update__(**kwargs)

        # store inference
        if self.store_flag:
            self.store()

        # print update
        logger.info(''.join([' - {} : {}'.format(ki, prints[ki]) for ki in list(prints.keys())]))
        
        # print tracing
        if tracemalloc.is_tracing():
            display_memory_usage(tracemalloc.take_snapshot())
            tracemalloc.clear_traces()

    def get_posterior(self):
        pass

    def make_plots(self):
        pass



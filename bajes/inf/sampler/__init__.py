#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import logging
logger = logging.getLogger(__name__)

import os
import signal
import tracemalloc
import numpy as np

from shutil import copyfile

from ...pipe import display_memory_usage
from ...pipe.utils import data_container

class SamplerBody(object):

    def __init__(self, engine, posterior,
                 ncheckpoint    = 0,
                 outdir         = './',
                 resume         = '/resume.pkl',
                 back           = '/backup.pkl',
                 seed           = None,
                 rank           = 0,
                 **kwargs):

        # engine tag
        self._engine = engine

        # output directory
        self.outdir  = outdir

        # resuming options
        self.resume         = resume
        self.back           = back
        self.lock_backup    = False

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

            # store random seed
            self.seed = seed

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
        # TODO: try container.inference, otherwise initialize new inference
        previous_inference  = container.inference
        for kw in list(previous_inference.keys()):
            logger.debug("Setting {} attribute ...".format(kw))
            self.__setattr__(kw, previous_inference[kw])

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

    def backup(self):

        if self.lock_backup == False:

            # check if resume is accessible
            access = self._check_resume()

            if access == 2:
                # if container is safe, rename it as backup
                logger.debug("Resume file is safe, updating backup ...")
                copyfile(self.outdir+self.resume, self.outdir+self.back)
            elif access == 1:
                # if container is safe,
                # but inference is not there, so there is not need to copy the resume
                logger.debug("Resume file is safe")
            else:
                # otherwise lock previous backup to the last safe iteration
                logger.warning("Unable to safely restore resume file. Locking backup file.")
                self.lock_backup = True

    def store(self):

        # backup of the previous container
        logger.debug("Backup of the previous container ...")
        self.backup()

        # save inference in pickle file
        logger.debug("Storing sampler in pickle ...")
        dc = data_container(self.outdir+self.resume)
        dc.store('inference', self.__getstate__())
        dc.save()

    def _first_store(self, posterior):
        # save posterior in pickle file
        logger.debug("Storing posterior ({}) in pickle ...".format(posterior))
        dc = data_container(self.outdir+self.resume)
        dc.store('tag', self._engine)
        dc.store('posterior', posterior)
        dc.save()

    def _restore_posterior(self):
        # restore posterior from pickle file
        logger.debug("Extracting posterior from existing file ...")
        dc = data_container(self.outdir+self.resume).load()
        return dc.posterior

    def _check_resume(self):
        # check container
        # return 0 if container is empty
        # return 1 if container contains posterior
        # return 2 if container contains inference and posterior
        logger.debug("Extracting posterior from existing file ...")
        container   = data_container(self.outdir+self.resume).load()
        _keys       = list(container.__dict__.keys())
        if 'posterior' in _keys and 'inference' in _keys:
            return 2
        elif 'posterior' in _keys:
            return 1
        else:
            return 0

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

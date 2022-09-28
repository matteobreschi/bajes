#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
import sys
import os

# logger
import logging
logger = logging.getLogger(__name__)

from . import ensure_dir, init_sampler, init_model

def print_header(engine, nprocs):

    import numpy as np
    import pathlib
    from bajes import __version__, __path__, __githash__

    logger.info("> bajes, Bayesian Jenaer Software")
    logger.info("* VERSION : {}".format(__version__))
    logger.info("* PATH    : {}".format(__path__[0]))
    logger.info("* GITHASH : {}".format(__githash__))
    logger.info("* ENGINE  : {}".format(engine.upper()))
    logger.info("> Running inference with {} processes".format(nprocs))

def finalize(inference):

    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        logger.warning("Unable to import matplotlib. Output plots will not be generated.")
        pass

    # get posterior samples and produces plots
    logger.info("Saving posterior samples ...")
    inference.get_posterior()
    inference.make_plots()
    logger.info("Sampling done.")

def run_main(opts, pool=None, close_pool=None):

    # initialize random seed
    import numpy, random

    if opts.seed == None:
        import time
        opts.seed = int(time.time())

    random.seed(opts.seed)
    numpy.random.seed(opts.seed)

    # header
    print_header(opts.engine, opts.nprocs)

    # get model
    pr, lk = init_model(opts)

    # initialize sampler
    inference = init_sampler([pr,lk], pool, opts)

    # # delete inputs
    # del opts

    # running sampler
    logger.info("Running sampling ...")
    inference.run()

    # close parallel pool
    if (close_pool is not None) and (pool is not None):
        close_pool(pool)

    # stop memory tracing, if needed
    if opts.trace_memory:
        tracemalloc.stop()

    # produce posteriors
    finalize(inference)

def run_main_mpi(opts, Pool):

    # initialize random seed
    # note that this step is common to all processes
    import numpy, random

    if opts.seed == None:
        # different workers might arrive at this point at different times
        # the best option would be to share master seed to workers
        # on the other hand if you end up here you're not specifying the random seed
        # meaning that you dont care
        import time
        opts.seed = int(time.time())

    random.seed(opts.seed+Pool.rank)
    numpy.random.seed(opts.seed+Pool.rank)

    # open pool
    # after this point all working processes are gathered
    with Pool as pool:

        # if not master, wait for a command from master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        # header
        print_header(opts.engine, opts.nprocs)
        logger.info("> MPI world initisalized")

        # initialize likelihood, prior and sampler
        pr, lk      = init_model(opts)

        # initialize sampler
        inference   = init_sampler([pr, lk], pool, opts)

        # running sampler
        logger.info("Running sampling ...")
        inference.run()

        # produce posteriors
        finalize(inference)

def run_main_mpi_ultranest(opts, rank, size):

    # NOTE: the inference is initialized in each MPI task,
    # in order to activate the multiprocessing pool (in each task, if needed)
    # Then the processes are joined together by ultranest routines, during the run()

    # initialize random seed
    # note that this step is common to all processes
    import numpy, random

    if opts.seed == None:
        # different workers might arrive at this point at different times
        # the best option would be to share master seed to workers
        # on the other hand if you end up here you're not specifying the random seed
        # meaning that you dont care
        import time
        opts.seed = int(time.time())

    random.seed(opts.seed+rank)
    numpy.random.seed(opts.seed+rank)

    # estimate cpu_per_task
    cpu_per_task = int(opts.nprocs)//size

    # check threading
    if cpu_per_task > 1:
        # activate multiprocessing pool for each MPI task
        logger.info("Activating multithreads pool on rank {} with {} threads.".format(rank, cpu_per_task))
        from . import initialize_mthr_pool
        pool, close_pool_mthr = initialize_mthr_pool(cpu_per_task)
    else:
        # only MPI
        pool = None
        close_pool_mthr = None

    # if master, print
    if rank == 0:
        if opts.debug:  logger.debug("Using logger with debugging mode")
        print_header(opts.engine, opts.nprocs)
        logger.info("> MPI world initisalized")

    # initialize likelihood, prior and sampler
    pr, lk      = init_model(opts)
    inference   = init_sampler([pr, lk], pool, opts, None, rank=rank)

    # running sampler
    logger.info("Running sampling ...")
    inference.run()

    # produce posteriors
    if rank == 0:
        finalize(inference)

    # close multiprocessing pools
    if pool is not None:
        close_pool_mthr(pool)

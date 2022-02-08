#!/usr/bin/env python
from __future__ import division, unicode_literals
import sys
import os
import logging

from bajes.pipe import set_logger, ensure_dir, init_sampler, parse_core_options

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

def print_header(logger, tags, engine, nprocs, p_tag=False):

    import numpy as np
    import pathlib
    from bajes import __version__, __path__, __githash__

    # set printing tag
    print_tag = ''
    for ti in tags:
        print_tag = print_tag + ti
        if ti != tags[-1]:
            print_tag = print_tag + '+'

    logger.info("> bajes, Bayesian Jenaer Software")
    logger.info("* VERSION : {}".format(__version__))
    logger.info("* PATH    : {}".format(__path__[0]))
    logger.info("* GITHASH : {}".format(__githash__))
    logger.info("* SIGNAL  : {}".format(print_tag.upper()))
    logger.info("* ENGINE  : {}".format(engine.upper()))
    if p_tag:
        logger.info("> Running bajes parallel core with {} processes".format( nprocs))
    else:
        logger.info("> Running bajes core with {} processes".format( nprocs))

def init_core(opts):

    import numpy as np

    # all models have distance and time_shift,
    # choose the smaller bounds
    if len(opts.dist_min) == 1:
        opts.dist_min       = opts.dist_min[0]
    elif len(opts.dist_min) == 0:
        opts.dist_min       = None
    else:
        opts.dist_min       = np.max(opts.dist_min)

    if len(opts.dist_max) == 1:
        opts.dist_max       = opts.dist_max[0]
    elif len(opts.dist_max) == 0:
        opts.dist_max       = None
    else:
        opts.dist_max       = np.min(opts.dist_max)

    if not opts.marg_time_shift:

        if len(opts.time_shift_min) == 1:
            opts.time_shift_min = opts.time_shift_min[0]
        elif len(opts.time_shift_min) == 0:
            opts.time_shift_min = None
        else:
            opts.time_shift_min = np.max(opts.time_shift_min)

        if len(opts.time_shift_max) == 1:
            opts.time_shift_max = opts.time_shift_max[0]
        elif len(opts.time_shift_max) == 0:
            opts.time_shift_max = None
        else:
            opts.time_shift_max = np.min(opts.time_shift_max)

    # get likelihood object and arguments
    from bajes.pipe import get_likelihood_and_prior
    like , prior, use_gw = get_likelihood_and_prior(opts)
    opts.__dict__['use_gw'] = use_gw

    # instance of Bayesian model
    from bajes.inf import Posterior
    post = Posterior(like=like, prior=prior)

    return opts, post

def finalize(inference, logger, rank=0):

    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        logger.warning("Unable to import matplotlib. Output plots will not be generated.")
        pass

    if rank == 0:
        # get posterior samples and produces plots
        logger.info("Saving posterior samples ...")
        inference.get_posterior()
        inference.make_plots()
        logger.info("Sampling done.")

if __name__ == "__main__":

    # parse input arguments
    opts,args = parse_core_options()

    # define global variables
    engine  = opts.engine
    nprocs  = opts.nprocs
    tracing = opts.trace_memory

    # start memory tracing, if requested
    if tracing:
        import tracemalloc
        tracemalloc.start(25)

    # make output directory and initialize logger
    opts.outdir = os.path.abspath(opts.outdir)
    ensure_dir(opts.outdir)
    if opts.debug:
        logger = set_logger(outdir=opts.outdir, level='DEBUG', silence=opts.silence)
        logger.debug("Using logger with debugging mode")
    else:
        logger = set_logger(outdir=opts.outdir, silence=opts.silence)

    # print header
    print_header(logger, opts.tags, engine, nprocs)

    # initialize multi-threading pool (if needed)
    if (engine != 'cpnest' and nprocs>1):
        from bajes.pipe import initialize_mthr_pool
        pool, close_pool   = initialize_mthr_pool(nprocs)
    else:
        pool = None
        close_pool = None

    # initialize posterior
    opts, post = init_core(opts)

    # initialize sampler
    inference = init_sampler(post, pool, opts)

    # delete inputs
    del opts
    del args

    # running sampler
    logger.info("Running sampling ...")
    inference.run()

    # close parallel pool, if needed
    if (engine != 'cpnest' and nprocs>1):
        close_pool(pool)

    # stop memory tracing, if needed
    if tracing:
        tracemalloc.stop()

    # produce posteriors
    finalize(inference, logger)
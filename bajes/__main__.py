#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import

import os, sys

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

if __name__ == "__main__":

    # parse options
    from .pipe import parse_main_options, set_logger, ensure_dir
    opts = parse_main_options()

    # start memory tracing, if requested
    if opts.trace_memory:
        import tracemalloc
        tracemalloc.start(opts.n_trace_memory)

    ###
    ### MPI RUN
    ###
    if opts.mpi:

        # set MPI variables
        try:

            os.environ["OMP_NUM_THREADS"]   = "1"
            os.environ["MKL_NUM_THREADS"]   = "1"
            os.environ["MKL_DYNAMIC"]       = "0"
            os.environ["MPI_PER_NODE"]      = "{}".format(opts.mpi_per_node)

            import mpi4py
            mpi4py.rc.threads       = False
            mpi4py.rc.recv_mprobe   = False

            from mpi4py import MPI
            size    = MPI.COMM_WORLD.Get_size()
            rank    = MPI.COMM_WORLD.Get_rank()

        except ImportError:
            raise ImportError("Unable to initialize MPI run. Cannot import mpi4py.")

        # switch between samplers
        # cpnest + MPI, not available
        if opts.engine == 'cpnest':
            raise AttributeError("MPI parallelisation not yet available with cpnest sampler.")

        # ultranest + MPI
        elif opts.engine == 'ultranest':

            # logger
            if opts.debug:
                logger = set_logger(outdir=opts.outdir, level='DEBUG', silence=opts.silence)
                logger.debug("Using logger with debugging mode")
            else:
                logger = set_logger(outdir=opts.outdir, silence=opts.silence)

            # execute run
            from .pipe.run import run_main_mpi_ultranest
            run_main_mpi_ultranest(opts, rank, size)

            # stop memory tracing, if requested
            if opts.trace_memory:
                tracemalloc.stop()

        # other samplers
        # i.e. the ones that work with the Pool object
        else:

            # initialize MPI pool
            from .pipe import initialize_mpi_pool
            Pool, close_pool = initialize_mpi_pool(mpi=MPI, comm=MPI.COMM_WORLD, fast_mpi=opts.fast_mpi)

            # logger and printings
            if Pool.rank == 0:

                # logger
                if opts.debug:
                    logger = set_logger(outdir=opts.outdir, level='DEBUG', silence=opts.silence)
                    logger.debug("Using logger with debugging mode")
                else:
                    logger = set_logger(outdir=opts.outdir, silence=opts.silence)

            else:

                # logger
                logger      = set_logger(outdir=opts.outdir, level='ERROR', silence=True)

                # delete inputs,
                # not needed for worker tasks
                del opts
                del args

            # execute run run
            from .pipe.run import run_main_mpi
            run_main_mpi(opts, Pool)
            # NOTE: at this stage only master is active

            # stop memory tracing, if requested
            if opts.trace_memory:
                tracemalloc.stop()

    ###
    ### MULTIPROCESSING RUN
    ###
    else:

        # make output directory
        opts.outdir = os.path.abspath(opts.outdir)
        ensure_dir(opts.outdir)

        # initialize logger
        if opts.debug:
            logger = set_logger(outdir=opts.outdir, level='DEBUG', silence=opts.silence)
            logger.debug("Using logger with debugging mode")
        else:
            logger = set_logger(outdir=opts.outdir, silence=opts.silence)

        # initialize multi-threading pool (if needed)
        if (opts.engine != 'cpnest' and opts.nprocs>1):
            from .pipe import initialize_mthr_pool
            pool, close_pool   = initialize_mthr_pool(opts.nprocs)
        else:
            pool = None
            close_pool = None

        # execute run
        from .pipe.run import run_main
        run_main(opts, pool=pool, close_pool=close_pool)

        # stop memory tracing, if requested
        if opts.trace_memory:
            tracemalloc.stop()

from __future__ import division, unicode_literals, absolute_import

class BajesMPIError(Exception):
    pass

def _check_mpi(is_mpi):

    multi_task  = False
    is_mpi      = bool(is_mpi)
    rank        = 0
    size        = 1

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if size > 1 :
            multi_task = True
    except Exception:
        pass

    if is_mpi != multi_task:

        if is_mpi == False and multi_task == True:
            logger.error("Revealed active MPI routine, please include the additional flag --mpi to the command string to use MPI parallelization.")
            raise BajesMPIError("Revealed active MPI routine, please include the additional flag --mpi to the command string to use MPI parallelization.")

        elif is_mpi == True and multi_task == False:
            logger.error("Requested MPI parallelization but only one task is active, please make sure you are using an MPI executer (i.e. mpiexec or mpirun) with the correct settings.")
            raise BajesMPIError("Requested MPI parallelization but only one task is active, please make sure you are using an MPI executer (i.e. mpiexec or mpirun) with the correct settings.")

    return rank, size

def _head(logger, engine, nprocs):

    from . import __version__, __path__, __githash__
    logger.info("> bajes, Bayesian Jenaer Software")
    logger.info("* VERSION : {}".format(__version__))
    logger.info("* PATH    : {}".format(__path__[0]))
    logger.info("* GITHASH : {}".format(__githash__))
    logger.info("* ENGINE  : {}".format(engine.upper()))
    logger.info("* PROCS   : {}".format(nprocs))

def main():

    import os
    import sys
    import importlib

    try:
        import configparser
    except ImportError:
        import ConfigParser as configparser

    from .inf import Prior, Parameter, Likelihood, Sampler
    from .pipe import parse_main_options, ensure_dir, set_logger

    # parse input
    opts, args = parse_main_options()

    # retreive absolute paths
    opts.prior  = os.path.abspath(opts.prior)
    opts.like   = os.path.abspath(opts.like)
    opts.outdir = os.path.abspath(opts.outdir)

    # ensure output directory
    ensure_dir(opts.outdir)

    # set logger
    if opts.debug:
        logger = set_logger(outdir=opts.outdir, level='DEBUG', silence=opts.silence)
        logger.debug("Using logger with debugging mode")
    else:
        logger = set_logger(outdir=opts.outdir, silence=opts.silence)

    _head(logger, opts.engine, opts.nprocs)

    # get likelihood module
    logger.info("Importing likelihood module ...")
    sys.path.append(os.path.dirname(opts.like))
    like_module = importlib.import_module(os.path.basename(opts.like).split('.')[0])

    # initialize likelihood
    lk = Likelihood(func = like_module.log_like)

    # parse prior
    logger.info("Parsing prior options ...")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.sections()
    config.read(opts.prior)

    # get parameters
    names       = [ni[0] for ni in list(config.items()) if ni[0] != 'DEFAULT']
    _checked    = ['min', 'max', 'periodic', 'prior', 'func', 'func_kwarg', 'interp_kwarg']
    params      = []

    for ni in names:

        args = dict(config[ni])
        list_args_keys = list(args.keys())

        # check lower bound
        if 'min' in list_args_keys:
             args['min'] = float(args['min'])
        else:
            logger.error("Unable to set prior for {} parameter, unspecified lower bound.".format(ni))
            raise AttributeError("Unable to set prior for {} parameter, unspecified lower bound.".format(ni))

        # check upper bound
        if 'max' in list_args_keys:
             args['max'] = float(args['max'])
        else:
            logger.error("Unable to set prior for {} parameter, unspecified upper bound.".format(ni))
            raise AttributeError("Unable to set prior for {} parameter, unspecified upper bound.".format(ni))

        # check bounds
        if 'periodic' in list_args_keys:
            args['periodic'] = int(args['periodic'])

        # check prior string/function
        if 'prior' in list_args_keys:
            logger.info("Setting {} with {} prior in range [{:.3g},{:.3g}]...".format(ni, args['prior'], args['min'], args['max']))
            if args['prior'] == 'custom':
                args['func'] = getattr(like_module, 'log_prior_{}'.format(ni))
        else:
            logger.info("Setting {} with uniform prior in range [{:.3g},{:.3g}] ...".format(ni, args['min'], args['max']))

        # set number of grid point for prior interpolation, if needed
        args['interp_kwarg'] = {'ngrid': opts.priorgrid}

        # check others
        for ki in list_args_keys:
            if ki not in _checked:
                args[ki] = float(args[ki])

        # append parameter to list
        params.append(Parameter(name = ni , **args))

    # initialize prior
    logger.info("Initializing prior distribution ...")
    pr = Prior(params)

    # ensure nwalk is even
    if opts.nwalk%2 != 0 :
        opts.nwalk += 1

    # set sampler kwargs
    kwargs = {  'nlive':        opts.nlive,
                'tolerance':    opts.tolerance,
                'maxmcmc':      opts.maxmcmc,
                'poolsize':     opts.poolsize,
                'minmcmc':      opts.minmcmc,
                'maxmcmc':      opts.maxmcmc,
                'nbatch':       opts.nbatch,
                'nwalk':        opts.nwalk,
                'nburn':        opts.nburn,
                'tmax':         opts.tmax,
                'nout':         opts.nout,
                'nact':         opts.nact,
                'dkl':          opts.dkl,
                'z_frac':       opts.z_frac,
                'ntemps':       opts.ntemps,
                'nprocs':       opts.nprocs,
                'seed':         opts.seed,
                'ncheckpoint':  opts.ncheck,
                'outdir':       opts.outdir,
                'nprocs':       opts.nprocs,
                'proposals_kwargs' : {'use_gw': False, 'use_slice': opts.use_slice}
                }

    # check MPI
    kwargs['rank'], size = _check_mpi(opts.mpi)

    # check for cpnest
    if opts.engine == 'cpnest':
        # set opts.nprocs to None in order to avoid parallel pool
        # cpnest treats parallel processes internally
        opts.nprocs = None
        if opts.mpi :
            logger.error("MPI parallelization not available with cpnest.")
            raise BajesMPIError("MPI parallelization not available with cpnest.")
        opts.mpi = False
        Pool, close_pool = None, None

    # set parallel pool if needed
    if opts.mpi:

        if opts.engine == 'ultranest':
            # ultranest has a customized MPI interface
            Pool, close_pool = None, None
        else:
            from .pipe import initialize_mpi_pool
            Pool, close_pool = initialize_mpi_pool(opts.nprocs)

    else:

        if opts.nprocs == None: opts.nprocs = 1
        if opts.nprocs >= 2:
            from .pipe import initialize_mthr_pool
            Pool, close_pool = initialize_mthr_pool(opts.nprocs)
        else:
            Pool, close_pool = None, None

    # checking sampler
    logger.info("Initializing sampler ...")
    from .inf import __known_samplers__
    if opts.engine not in __known_samplers__:
        logger.error("Invalid string for engine. Please use one of the following: {}".format(__known_samplers__))
        raise AttributeError("Invalid string for engine. Please use one of the following: {}".format(__known_samplers__))

    # running
    if Pool == None:

        # set multithreading pools for each MPI process
        if opts.engine == 'ultranest' and opts.mpi:

            threads_per_node = int(opts.nprocs)//size

            if threads_per_node < 2 :
                logger.debug("Estimated number of threads per MPI process is lower than 2, disabling vectorization for ultranest.")
            else:
                logger.debug("Running {} MPI processes with {} threads per task.".format(size, threads_per_node))
                from .pipe import initialize_mthr_pool
                Pool, close_pool = initialize_mthr_pool(threads_per_node)
                kwargs['pool'] = Pool

        sampler = Sampler(opts.engine, [lk, pr], **kwargs)
        sampler.run()
        sampler.get_posterior()
        sampler.make_plots()

    else:

        with Pool as pool:

            if opts.mpi:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)

            kwargs['pool'] = pool
            sampler = Sampler(opts.engine, [lk, pr], **kwargs)
            sampler.run()
            sampler.get_posterior()
            sampler.make_plots()

            close_pool(pool)

    logger.info("Sampling done.")

if __name__ == "__main__":
    main()

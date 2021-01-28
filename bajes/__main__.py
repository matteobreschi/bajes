from __future__ import division, unicode_literals, absolute_import

def _head(logger, engine, nprocs):

    import os
    import pathlib
    from . import __version__, __path__
    
    # look for git hash
    git_hash = 'UNKNOWN'
    for pi in __path__:

        dir_path = os.path.abspath(os.path.join(pi,'..'))
        _ld = os.listdir(dir_path)
        
        if '.git' in _ld:
            git_dir = pathlib.Path(dir_path) / '.git'
            with (git_dir / 'HEAD').open('r') as head:
                ref = head.readline().split(' ')[-1].strip()
            with (git_dir / ref).open('r') as git_hash:
                git_hash = git_hash.readline().strip()
            break

    logger.info("> bajes, Bayesian Jenaer Software")
    logger.info("* VERSION : {}".format(__version__))
    logger.info("* PATH    : {}".format(__path__[0]))
    logger.info("* GITHASH : {}".format(git_hash))
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
    opts.prior = os.path.abspath(opts.prior)
    opts.like = os.path.abspath(opts.like)
    opts.outdir = os.path.abspath(opts.outdir)

    # ensure output directory
    ensure_dir(opts.outdir)

    # set logger
    logger = set_logger(outdir=opts.outdir, label='bajes', silence=opts.silence)
    _head(logger, opts.engine, opts.nprocs)

    # get likelihood module
    logger.info("Importing likelihood module ...")
    sys.path.append(os.path.dirname(opts.like))
    like_module = os.path.basename(opts.like).split('.')[0]
    like_module = importlib.import_module(like_module)

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

        # check others
        for ki in list_args_keys:
            if ki not in _checked:
                args[ki] = float(args[ki])

        params.append(Parameter(name = ni , **args))

    # initialize likelihood
    logger.info("Initializing prior distribution ...")
    pr = Prior(params)
  
    # ensure nwalk is even
    if opts.nwalk%2 != 0 :
        opts.nwalk += 1

    kwargs = {  'nlive':        opts.nlive,
                'tolerance':    opts.tolerance,
                'maxmcmc':      opts.maxmcmc,
                'poolsize':     opts.poolsize,
                'minmcmc':      opts.minmcmc,
                'nbatch':       opts.nbatch,
                'nwalk':        opts.nwalk,
                'nburn':        opts.nburn,
                'nout':         opts.nout,
                'nact':         opts.nact,
                'ntemps':       opts.ntemps,
                'nprocs':       opts.nprocs,
                'seed':         opts.seed,
                'ncheckpoint':  opts.ncheck,
                'outdir':       opts.outdir,
                'nprocs':       opts.nprocs,
                'proposals_kwargs' : {'use_gw': False, 'use_slice': opts.use_slice}
                }

    # check for cpnest
    if opts.engine == 'cpnest':
        if opts.mpi :
            logger.warning("MPI parallelization not available with cpnest, turning the flag off. You are currently running many copies of the same routine.")
        opts.mpi = False

    # initialize parallel pool
    if opts.mpi:
        from .pipe import initialize_mpi_pool
        Pool, close_pool = initialize_mpi_pool(opts.nprocs)
    else:
        if opts.nprocs == None: opts.nprocs = 1
        if opts.nprocs >= 2:
            from .pipe import initialize_mthr_pool
            Pool, close_pool = initialize_mthr_pool(opts.nprocs)
        else:
            Pool, close_pool = None, None

    # initialize sampler
    logger.info("Initializing sampler ...")
    if opts.engine not in ['mcmc', 'ptmcmc', 'cpnest', 'nest', 'dynest']:
        logger.error("Invalid string for engine. Please use one of the following engines: 'cpnest' , 'nest' , 'dynest' , 'mcmc', 'ptmcmc'.")
        raise AttributeError("Invalid string for engine. Please use one of the following engines: 'cpnest' , 'nest' , 'dynest' , 'mcmc', 'ptmcmc'.")

    # running
    if Pool == None:
        
        sampler = Sampler(opts.engine, [lk, pr], **kwargs)
        sampler.run()
        sampler.get_posterior()

    else:

        with Pool as pool:

            if opts.mpi:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)

            kwargs['pool'] = pool
            kwargs['pool'] = pool
            sampler = Sampler(opts.engine, [lk, pr], **kwargs)
            sampler.run()
            sampler.get_posterior()

            close_pool(pool)

    logger.info("Sampling done.")

if __name__ == "__main__":
    
    main()

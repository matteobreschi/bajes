#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import os
import numpy as np

from scipy.linalg import expm, norm

try:
    import subprocess
except ImportError:
    pass

try:
    import pickle
except ImportError:
    import cPickle as pickle

# logger

import logging
logger = logging.getLogger(__name__)

def set_logger(label=None, outdir=None, level='INFO', silence=True):

    if label == None:
        label = 'bajes'

    if level.upper() == 'DEBUG':
        datefmt = '%m-%d-%Y-%H:%M:%S'
        fmt     = '[{}] [%(asctime)s.%(msecs)04d] %(levelname)s: %(message)s'.format(label)
    else:
        datefmt = '%m-%d-%Y-%H:%M'
        fmt     = '[{}] [%(asctime)s] %(levelname)s: %(message)s'.format(label)

    # initialize logger
    logger = logging.getLogger(label)
    logger.propagate = False
    logger.setLevel(('{}'.format(level)).upper())

    # set streamhandler
    if not silence:
        if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            stream_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(stream_handler)

    # set filehandler
    if outdir != None:
        if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            file_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(file_handler)

    return logger

# memory

def display_memory_usage(snapshot, limit=5):
    
    import tracemalloc
    
    # get snapshot information ignoring importlib and unknown packages
    snapshot = snapshot.filter_traces((tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                                       tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                                       tracemalloc.Filter(False, "<unknown>"),
                                       ))
    
    # display memory usage
    memory_lineno(snapshot)
    memory_traceback(snapshot)

def memory_traceback(snapshot, limit=1):
    
    # get snapshot statistics (as traceback)
    top_stats = snapshot.statistics('traceback')

    # pick the biggest memory block
    stat = top_stats[0]
    logger.info("Tracback most expensive call ({} memory blocks, {:.1f} KiB):".format(limit, stat.count, stat.size/1024.))
    for line in stat.traceback.format():
        logger.info("{}".format(line))

def memory_lineno(snapshot, limit=5):
    
    import linecache
    
    # get snapshot statistics (as line+number)
    top_stats = snapshot.statistics('lineno')

    # print summary
    logger.info("Memory usage summary ({}):".format(limit))
    for i, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        logger.info(">>> n.{} - {}:{}: {:.1f} KiB".format(i, filename, frame.lineno, stat.size / 1024.))
        logger.info("\t{}".format(linecache.getline(frame.filename, frame.lineno).strip()))
    
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.info(">>> Other {} activities were used:  {:.1f} KiB".format(len(other), size / 1024.))
    total = sum(stat.size for stat in top_stats)
    logger.info(">>> Total allocated size: {:.1f} KiB".format(total / 1024.))

# wrappers

def eval_func_tuple(f_args):
    return f_args[0](*f_args[1:])

def erase_init_wrapper(cls):
    cls.__init__ = None
    return cls

# geometry

def cart2sph(x,y,z):
    """ x, y, z :  ndarray coordinates
    """
    r       = np.sqrt(x**2 + y**2 + z**2)
    phi     = np.arctan2(y,x)
    theta   = np.arccos(z/r)

    # now we have: theta in [0 , pi] and phi in [- pi , + pi]
    # move to phi in [0 , 2pi]
    if phi < 0 :
        phi = 2*np.pi + phi

    return r, theta, (phi)%(2*np.pi)

def sph2cart(r,theta,phi):
    """ r, theta, phi :  ndarray coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def rotation_matrix(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

# quasi-bash

def ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass

def execute_bash(bash_command):
    try:
        # python3
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, error = process.communicate()
    except Exception:
        # python2
        os.system(bash_command)

# parallel pool

def close_pool_mpi(pool):
    'close processes before the end of the pool, with mpi4py'
    pool.close()

def initialize_mpi_pool(fast_mpi=False):
    from .utils.mpi import MPIPool
    pool = MPIPool(parallel_comms=fast_mpi)
    close_pool = close_pool_mpi
    return  pool, close_pool

def close_pool_mthr(pool):
    'close processes before the end of the pool, with multiprocessing'
    try:
        pool.close()
    except:
        pool.terminate()
    finally:
        pool.join()

def initialize_mthr_pool(nprocs):
    if nprocs == None:
        return None, None
    logger.info("Initializing parallel pool ...")
    from multiprocessing import Pool
    pool = Pool(nprocs-1)
    close_pool = close_pool_mthr
    return  pool, close_pool

# container

def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True

def save_container(path, kwargs):
    """
        Save dictionary of objects in data container,
        Args:
        - path : path string to outpuot
        - kwargs : dictionary of objects, the keys will define the arguments of the container
    """
    
    pkl_kwarg = {}
    for ki in list(kwargs.keys()):
        if is_picklable(kwargs[ki]):
            pkl_kwarg[ki] = kwargs[ki]
        else:
            logger.warning("Impossible to store {} object in data container, it is not picklable".format(ki))

    dc = data_container(path)
    for ki in list(pkl_kwarg.keys()):
        logger.debug("Storing {} object in data container".format(ki))
        dc.store(ki, pkl_kwarg[ki])
    dc.save()

class data_container(object):
    """
        Object for storing MCMC Inference class,
        It restores all the settings from previous iterations.
    """

    def __init__(self, filename):
        # initialize with a filename
        self.filename = filename

    def store(self, name, data):
        # include data objects in this class
        self.__dict__[name] = data

    def save(self):
        
        # check stored objects
        if os.path.exists(self.filename):
            _stored     = self.load()
            
            # join old and new data if the container is not empty
            if _stored is not None:
                _current        = list(self.__dict__.keys())
                _old            = {ki: _stored.__dict__[ki] for ki in list(_stored.__dict__.keys()) if ki not in _current}
                self.__dict__   = {**self.__dict__, **_old}
        
        # save objects into filename
        f = open(self.filename, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        # load from existing filename
        f = open(self.filename, 'rb')
        try:
            n = pickle.load(f)
            f.close()
            return n
        except Exception:
            return None

# pipeline/core methods

def parse_main_options():

    from .. import __version__, __doc__
    from ..inf import __known_samplers__
    import optparse as op
    
    usage   = "python -m bajes [options]\n"+"Version: bajes {}".format(__version__)
    parser=op.OptionParser(usage=usage, version=__version__, description="Description:\n"+__doc__)

    # Choose the engine
    parser.add_option('-p', '--prior',  dest='prior',       type='string',  help='path to prior file (configuration file)')

    # Choose the engine
    parser.add_option('-l', '--like',   dest='like',        type='string',  help='path to likelihood function (python file)')

    # Choose the engine
    parser.add_option('-s', '--sampler', dest='engine',      default='dynesty', type='string',  help='sampler engine name, {}'.format(__known_samplers__))

    # Parallelization option (only multiprocessing)
    parser.add_option('-n', '--nprocs',       dest='nprocs',    default=None,   type='int', help='number of parallel processes')

    # output
    parser.add_option('-o', '--outdir', default='./',       type='string',  dest='outdir',  help='output directory')

    # Nested sampling options
    parser.add_option('--nlive',        dest='nlive',       default=1024,   type='int',     help='[nest] number of live points')
    parser.add_option('--tol',          dest='tolerance',   default=0.1,    type='float',   help='[nest] evidence tolerance')
    parser.add_option('--maxmcmc',      dest='maxmcmc',     default=4096,   type='int',     help='[nest] maximum number of mcmc iterations')
    parser.add_option('--minmcmc',      dest='minmcmc',     default=32,     type='int',     help='[nest] minimum number of mcmc iterations')
    parser.add_option('--poolsize',     dest='poolsize',    default=2048,   type='int',     help='[nest] number of sample in the pool (cpnest)')
    parser.add_option('--nact',         dest='nact',        default=5,      type='int',     help='[nest] sub-chain safe factor (dynesty)')
    parser.add_option('--nbatch',       dest='nbatch',      default=512,    type='int',     help='[nest] number of live points for batch (dynesty-dyn)')
    parser.add_option('--dkl',          dest='dkl',         default=0.5,    type='float',   help='[nest] target KL divergence (ultranest)')
    parser.add_option('--z-frac',       dest='z_frac',      default=None,   type='float',   help='[nest] remaining Z fraction (ultranest)')
    
    # MCMC options
    parser.add_option('--nout',         dest='nout',        default=10000,  type='int',     help='[mcmc] number of posterior samples')
    parser.add_option('--nwalk',        dest='nwalk',       default=256,    type='int',     help='[mcmc] number of parallel walkers')
    parser.add_option('--nburn',        dest='nburn',       default=5000,    type='int',    help='[mcmc] numebr of burn-in iterations')
    parser.add_option('--ntemp',        dest='ntemps',      default=8,      type='int',     help='[mcmc] number of tempered ensambles (ptmcmc)')
    parser.add_option('--tmax',         dest='tmax',        default=None,   type='float',   help='[mcmc] maximum temperature scale, default inf (ptmcmc)')

    # Others
    parser.add_option('--priorgrid',    dest='priorgrid',   default=1000,   type='int',             help='number of nodes for prior interpolators (if needed)')
    parser.add_option('--use-slice',    dest='use_slice',   default=False,  action="store_true",    help='use slice proposal (emcee or cpnest)')
    parser.add_option('--checkpoint',   dest='ncheck',      default=0,      type='int',             help='number of periodic checkpoints')
    parser.add_option('--seed',         dest='seed',        default=None,   type='int',             help='seed for the pseudo-random generator')
    parser.add_option('--mpi',         dest='mpi',        default=False,  action="store_true",      help='use MPI parallelization')

    # logging
    parser.add_option('-v', '--verbose',          dest='silence',     default=True,  action="store_false",    help='activate stream handler, use this if you are running on terminal')
    parser.add_option('--debug',            dest='debug',       default=False,  action="store_true",    help='use debugging mode for logger')

    (opts,args) = parser.parse_args()
    return opts,args

def parse_core_options():

    from .. import __version__, __doc__
    from ..inf import __known_samplers__
    import optparse as op
    
    usage   = "bajes_core.py [options]"+"Version: bajes {}".format(__version__)
    parser=op.OptionParser(usage=usage, version=__version__, description="Description:\n"+__doc__)

    # KN/GW tag
    parser.add_option('--tag',          dest='tags',        type='string',  action="append",    default=[],    help='Tag for data messenger, i.e. gw or kn')

    # GPS time
    parser.add_option('--t-gps',        dest='t_gps',       type='float',   help='GPS time: for GW, center value of time axis; for KN, initial value of time axis')

    # Choose the engine
    parser.add_option('--engine',       dest='engine',      default='dynesty', type='string',  help='sampler engine name, {}'.format(__known_samplers__))

    # Prior grid interpolators
    parser.add_option('--priorgrid',    dest='priorgrid',   default=1000,   type='int',     help='number of nodes for prior interpolators (if needed)')

    # Nested sampling options
    parser.add_option('--nlive',        dest='nlive',       default=1024,   type='int',     help='number of live points')
    parser.add_option('--tol',          dest='tolerance',   default=0.1,    type='float',   help='evidence tolerance')
    parser.add_option('--maxmcmc',      dest='maxmcmc',     default=4096,   type='int',     help='maximum number of mcmc iterations')
    parser.add_option('--minmcmc',      dest='minmcmc',     default=32,     type='int',     help='minimum number of mcmc iterations')
    parser.add_option('--poolsize',     dest='poolsize',    default=2048,   type='int',     help='number of sample in the pool (cpnest)')
    parser.add_option('--nact',         dest='nact',        default=5,      type='int',     help='sub-chain safe factor (dynesty)')
    parser.add_option('--nbatch',       dest='nbatch',      default=512,    type='int',     help='number of live points for batch (dynesty-dyn)')
    parser.add_option('--dkl',          dest='dkl',         default=0.5,    type='float',   help='target KL divergence (ultranest)')
    parser.add_option('--z-frac',       dest='z_frac',      default=None,   type='float',   help='remaining Z fraction (ultranest)')
    
    # MCMC options
    parser.add_option('--nout',         dest='nout',        default=4000,   type='int',     help='number of posterior samples')
    parser.add_option('--nwalk',        dest='nwalk',       default=256,    type='int',     help='number of parallel walkers')
    parser.add_option('--nburn',        dest='nburn',       default=15000,  type='int',     help='numebr of burn-in iterations')
    parser.add_option('--ntemp',        dest='ntemps',      default=8,      type='int',     help='number of tempered ensambles (ptmcmc)')
    parser.add_option('--tmax',         dest='tmax',        default=None,   type='float',   help='maximum temperature scale, default inf (ptmcmc)')

    # Distance information
    parser.add_option('--dist-flag',    dest='dist_flag',       default='vol',  type='string',                      help='distance prior flag (options: vol, log, com, src)')
    parser.add_option('--dist-min',     dest='dist_min',        default=[],     type='float',   action="append",    help='lower distance prior bound')
    parser.add_option('--dist-max',     dest='dist_max',        default=[],     type='float',   action="append",    help='upper distance prior bound')

    # Time shift (from GPS time) information
    parser.add_option('--tshift-max',   dest='time_shift_max',  default=[],     type='float',   action="append",    help='upper time shift prior bound')
    parser.add_option('--tshift-min',   dest='time_shift_min',  default=[],     type='float',   action="append",    help='lower time shift prior bound')

    # Fixed parameter options
    parser.add_option('--fix-name',     dest='fixed_names',     default=[],     action="append", type='string',     help='names of fixed params')
    parser.add_option('--fix-value',    dest='fixed_values',    default=[],     action="append", type='float',      help='values of fixed params')

    # Parallelization option
    parser.add_option('--nprocs',       dest='nprocs',          default=None,   type='int',             help='number of processes in the pool')
    parser.add_option('--mpi-per-node', dest='mpi_per_node',    default=None,   type='int',             help='number of MPI processes per node')
    parser.add_option('--fast-mpi',     dest='fast_mpi',        default=False,  action="store_true",    help='enable fast MPI communication')

    # Others
    parser.add_option('--use-slice',        dest='use_slice',   default=False,  action="store_true",    help='use slice proposal (emcee or cpnest)')
    parser.add_option('--checkpoint',       dest='ncheck',      default=0,      type='int',             help='number of periodic checkpoints')
    parser.add_option('--seed',             dest='seed',        default=None,   type='int',             help='seed for the pseudo-random chain')
    parser.add_option('--debug',            dest='debug',       default=False,  action="store_true",    help='use debugging mode for logger')
    parser.add_option('--verbose',          dest='silence',     default=True,  action="store_false",    help='activate stream handler, use this if you are running on terminal')
    parser.add_option('--tracing',            dest='trace_memory',       default=False,  action="store_true",    help='keep track of memory usage')
    parser.add_option('-o', '--outdir',     default=None,       type='string',  dest='outdir',          help='directory for output')

    #
    # GW OPTIONS
    #

    # Data and PSDs information
    parser.add_option('--ifo',      dest='ifos',        type='string',  action="append", default=[],    help='IFO tag, i.e. H1, L1, V1, K1, G1')
    parser.add_option('--strain',   dest='strains',     type='string',  action="append", default=[],    help='path to strain data')
    parser.add_option('--asd',      dest='asds',        type='string',  action="append", default=[],    help='path to ASD data')

    # Optional, calibration envelopes
    parser.add_option('--spcal',    dest='spcals',      type='string',      default=[],     action="append",    help='path to calibration envelope')
    parser.add_option('--nspcal',   dest='nspcal',      type='int',         default=0,      help='number of spectral calibration nodes')

    # Time series information
    parser.add_option('--f-min',    dest='f_min',       default=None,   type='float',   help='minimum frequency [Hz]')
    parser.add_option('--f-max',    dest='f_max',       default=None,   type='float',   help='maximum frequency [Hz]')
    parser.add_option('--srate',    dest='srate',       default=None,   type='float',   help='sampling rate [Hz]')
    parser.add_option('--seglen',   dest='seglen',      default=None,   type='float',   help='length of the segment [sec]')
    parser.add_option('--lmax',     dest='lmax',        default=0,      type='int',     help='higher order mode for GW template')
    parser.add_option('--alpha',    dest='alpha',       default=None,   type='float',   help='alpha parameter of the Tukey window')

    # Waveform model
    parser.add_option('--approx',   dest='approx',      default=None,   type='string',  help='gravitational-wave approximant')

    # Prior flags
    parser.add_option('--data-flag',    dest='data_flag',           default=None,           type='string',  help='spin prior flag')
    parser.add_option('--spin-flag',    dest='spin_flag',           default='no-spins',     type='string',  help='spin prior flag')
    parser.add_option('--tidal-flag',   dest='lambda_flag',         default='no-tides',     type='string',  help='tidal prior flag')

    # Prior bounds
    parser.add_option('--mc-min',       dest='mchirp_min',      default=None,   type='float',   help='lower mchirp prior bound')
    parser.add_option('--mc-max',       dest='mchirp_max',      default=None,   type='float',   help='upper mchirp prior bound')
    parser.add_option('--q-max',        dest='q_max',           default=None,   type='float',   help='upper mass ratio prior bound')
    parser.add_option('--mass-max',     dest='mass_max',       default=None,   type='float',   help='upper mass component prior bound')
    parser.add_option('--mass-min',     dest='mass_min',       default=None,   type='float',   help='lower mass component prior bound')
    parser.add_option('--spin-max',     dest='spin_max',        default=None,   type='float',   help='upper spin prior bound')
    parser.add_option('--lambda-min',   dest='lambda_min',      default=None,   type='float',   help='lower tidal prior bound')
    parser.add_option('--lambda-max',   dest='lambda_max',      default=None,   type='float',   help='upper tidal prior bound')

    # Extra parameters
    parser.add_option('--use-energy-angmom',    dest='ej_flag',     default=False,  action="store_true",    help='include energy and angular momentum parameters')
    parser.add_option('--use-eccentricity',     dest='ecc_flag',    default=False,  action="store_true",    help='include energy and angular momentum parameters')
    parser.add_option('--e-min',    dest='e_min',   type='float',   default=None,   help='lower energy prior bound')
    parser.add_option('--e-max',    dest='e_max',   type='float',   default=None,   help='upper energy prior bound')
    parser.add_option('--j-min',    dest='j_min',   type='float',   default=None,   help='lower angular momentum prior bound')
    parser.add_option('--j-max',    dest='j_max',   type='float',   default=None,   help='upper angular momentum prior bound')
    parser.add_option('--ecc-min',  dest='ecc_min', type='float',   default=None,   help='lower eccentricity prior bound')
    parser.add_option('--ecc-max',  dest='ecc_max', type='float',   default=None,   help='upper eccentricity prior bound')

    # Optional, marginalize over phi_ref and/or time_shift
    parser.add_option('--marg-phi-ref',         dest='marg_phi_ref',        default=False,  action="store_true",   help='phi-ref marginalization flag')
    parser.add_option('--marg-time-shift',      dest='marg_time_shift',     default=False,  action="store_true",   help='time-shift marginalization flag')

    # Optional, number of PSD weights
    parser.add_option('--psd-weights',   dest='nweights',         default=0,     type='int',  help='number of PSD weight parameters per IFO, default 0')

    # GWBinning options
    parser.add_option('--use-binning',  dest='binning',     default=False,  action="store_true",    help='frequency binning flag')
    parser.add_option('--fiducial',     dest='fiducial',    default=None,   type='string',  help='path to parameters file for gwbinning')

    #
    # KN OPTIONS
    #

    # Data & Components information
    parser.add_option('--comp',         dest='comps',       type='string',  action="append", default=[],    help='Name of shell component(s) for lightcurve estimation')
    parser.add_option('--mag-folder',   dest='mag_folder',  type='string',  default=None,    help='Path to magnitudes data folder')

    # Photometric bands information
    parser.add_option('--band',         dest='bands',       type='string',  action="append",    default=[], help='Name of photometric bands used in the data')
    parser.add_option('--lambda',       dest='lambdas',     type='float',   action="append",    default=[], help='Wave-length of photometric bands used in the data [nm]')
    parser.add_option('--use-dereddening',  dest='dered',   default=True,  action="store_true",    help='apply deredding to given data filters')

    # Prior bounds
    parser.add_option('--mej-max',      dest='mej_max',     type='float',   action="append",    default=[], help='Upper bounds for ejected mass parameters')
    parser.add_option('--mej-min',      dest='mej_min',     type='float',   action="append",    default=[], help='Lower bounds for ejected mass parameters')
    parser.add_option('--vel-max',      dest='vel_max',     type='float',   action="append",    default=[], help='Upper bounds for velocity parameters')
    parser.add_option('--vel-min',      dest='vel_min',     type='float',   action="append",    default=[], help='Lower bounds for velocity parameters')
    parser.add_option('--opac-max',     dest='opac_max',    type='float',   action="append",    default=[], help='Upper bounds for opacity parameters')
    parser.add_option('--opac-min',     dest='opac_min',    type='float',   action="append",    default=[], help='Lower bounds for opacity parameters')

    # Heating factor information
    parser.add_option('--log-eps0',     dest='log_eps_flag',    default=False,  action="store_true",   help='log-epsilon0 prior flag')
    parser.add_option('--eps-max',      dest='eps_max',     type='float',   default=None,       help='Upper bounds for heating factor parameter')
    parser.add_option('--eps-min',      dest='eps_min',     type='float',   default=None,       help='Lower bounds for heating factor parameter')

    # Extra heating rate coefficients
    parser.add_option('--sample-heating',   dest='heat_sampling',    default=False,  action="store_true",   help='Include extra heating coefficients in sampling, default False')
    parser.add_option('--heat-alpha',       dest='heating_alpha',     type='float',   default=1.3,          help='alpha coefficient for heating rate (default 1.3)')
    parser.add_option('--heat-time',        dest='heating_time',     type='float',   default=1.3,           help='time coefficient for heating rate (default 1.3)')
    parser.add_option('--heat-sigma',       dest='heating_sigma',     type='float',   default=0.11,         help='sigma coefficient for heating rate (default 0.11)')

    # Integrators properties
    parser.add_option('--nvel',         dest='n_v',         type='int',     default=400,        help='Number of elements in velocity array, default 400')
    parser.add_option('--vel-min-grid', dest='vgrid_min',   type='float',   default=1.e-7,      help='Lower limit for velocity integration, default 1e-7')
    parser.add_option('--ntime',        dest='n_t',         type='int',     default=400,        help='Number of elements in time array, default 400')
    parser.add_option('--t-start-grid', dest='init_t',      type='float',   default=1.,         help='Initial value of time axis for model evaluation, default 1s')
    parser.add_option('--t-scale',      dest='t_scale',     type='string',  default='linear',   help='Scale of time axis: linear, log or mixed')

    (opts,args) = parser.parse_args()
    return opts,args


def init_sampler(posterior, pool, opts, proposals=None, rank=0):
    
    from ..inf import Sampler

    # ensure nwalk is even
    if opts.nwalk%2 != 0 :
        opts.nwalk += 1

    kwargs = {  'nlive':        opts.nlive,
                'tolerance':    opts.tolerance,
                'maxmcmc':      opts.maxmcmc,
                'poolsize':     opts.poolsize,
                'minmcmc':      opts.minmcmc,
                'maxmcmc':      opts.maxmcmc,
                'nbatch':       opts.nbatch,
                'nwalk':        opts.nwalk,
                'nburn':        opts.nburn,
                'nout':         opts.nout,
                'nact':         opts.nact,
                'dkl':          opts.dkl,
                'tmax':         opts.tmax,
                'z_frac':       opts.z_frac,
                'ntemps':       opts.ntemps,
                'nprocs':       opts.nprocs,
                'pool':         pool,
                'seed':         opts.seed,
                'ncheckpoint':  opts.ncheck,
                'outdir':       opts.outdir,
                'proposals':    proposals,
                'rank':         rank,
                'proposals_kwargs' : {'use_gw': opts.use_gw, 'use_slice': opts.use_slice}
                }

    return Sampler(opts.engine, posterior, **kwargs)

def init_proposal(engine, post, use_slice=False, use_gw=False, maxmcmc=4096, minmcmc=32, nact=5.):
    
    logger.info("Initializing proposal methods ...")

    if engine == 'emcee':
        from ..inf.sampler.emcee import initialize_proposals
        return initialize_proposals(post.like, post.prior, use_slice=use_slice, use_gw=use_gw)

    elif engine == 'ptmcmc':
        from ..inf.sampler.ptmcmc import initialize_proposals
        return initialize_proposals(post.like, post.prior, use_slice=use_slice, use_gw=use_gw)

    elif engine == 'cpnest':
        from ..inf.sampler.cpnest import initialize_proposals
        return initialize_proposals(post, use_slice=use_slice, use_gw=use_gw)

    elif 'dynesty' in engine:
        from ..inf.sampler.dynesty import initialize_proposals
        return initialize_proposals(maxmcmc=maxmcmc, minmcmc=minmcmc, nact=nact)

    elif engine == 'ultranest':
        return None

def get_likelihood_and_prior(opts):

    # get likelihood objects
    likes = []
    priors = []
    use_gw = False

    for ti in opts.tags:

        if ti == 'gw':

            # select GW likelihood, with or without binning
            if opts.binning:
                from .utils.binning import GWBinningLikelihood as GWLikelihood
            else:
                from .utils.model import GWLikelihood

            # read arguments for likelihood
            l_kwas, pr = initialize_gwlikelihood_kwargs(opts)
            likes.append(GWLikelihood(**l_kwas))
            priors.append(pr)

            # set use_gw flag for proposal
            use_gw = True

        elif ti == 'kn':

            # select KN likelihood
            from .utils.model import KNLikelihood

            # read arguments for likelihood
            l_kwas, pr = initialize_knlikelihood_kwargs(opts)
            likes.append(KNLikelihood(**l_kwas))
            priors.append(pr)

        else:

            logger.error("Unknown tag {} for likelihood initialization. Please use gw, kn or a combination.".format(opts.tags[0]))
            raise ValueError("Unknown tag {} for likelihood initialization. Please use gw, kn or a combination.".format(opts.tags[0]))

    logger.info("Initializing likelihood ...")

    # reduce to single likelihood object (single or joined)
    if len(opts.tags) == 0:
        logger.error("Unknown tag for likelihood initialization. Please use gw, kn or a combination.")
        raise ValueError("Unknown tag for likelihood initialization. Please use gw, kn or a combination.")
    
    elif len(opts.tags) == 1:
        l_obj   = likes[0]
        p_obj   = priors[0]

    else:

        from ..inf.prior import JointPrior
        from ..inf.likelihood import JointLikelihood

        l_kwas              = {}
        l_kwas['likes']     = likes
        
        p_obj   = JointPrior(priors=priors, prior_grid=opts.priorgrid)
        l_obj   = JointLikelihood(**l_kwas)

    # save prior and likelihood in pickle
    cont_kwargs = {'prior': p_obj, 'like': l_obj}
    save_container(opts.outdir+'/inf.pkl', cont_kwargs)

    return l_obj , p_obj, use_gw

def initialize_knlikelihood_kwargs(opts):

    from ..obs.kn.filter import Filter

    # initial check
    if (len(opts.comps) != len(opts.mej_min)) or (len(opts.comps) != len(opts.mej_max)):
        logger.error("Number of components does not match the number of ejected mass bounds. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of components does not match the number of ejected mass bounds. Please give in input the same number of arguments in the respective order.")
    if (len(opts.comps) != len(opts.vel_min)) or (len(opts.comps) != len(opts.vel_max)):
        logger.error("Number of components does not match the number of velocity bounds. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of components does not match the number of velocity bounds. Please give in input the same number of arguments in the respective order.")
    if (len(opts.comps) != len(opts.opac_min)) or (len(opts.comps) != len(opts.opac_max)):
        logger.error("Number of components does not match the number of opacity bounds. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of components does not match the number of opacity bounds. Please give in input the same number of arguments in the respective order.")
    if opts.time_shift_min == None:
        opts.time_shift_min = -opts.time_shift_max

    # initialize wavelength dictionary for photometric bands
    lambdas = {}
    if len(opts.lambdas == 0):

        # if lambdas are not given use the standard ones
        from ..obs.kn import __photometric_bands__ as ph_bands
        for bi in opts.bands:
            if bi in list(ph_bands.keys()):
                lambdas[bi] = ph_bands[bi]
            else:
                logger.error("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))
                raise ValueError("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))

    else:
        # check bands
        if len(opts.bands) != len(opts.lambdas):
            logger.error("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")
            raise ValueError("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")

        for bi,li in zip(opts.bands, opts.lambdas):
            lambdas[bi] = li

    # initialize likelihood keyword arguments
    l_kwargs = {}
    l_kwargs['comps']       = opts.comps
    l_kwargs['filters']     = Filter(opts.mag_folder, lambdas, dered=opts.dered)
    l_kwargs['v_min']       = opts.vgrid_min
    l_kwargs['n_v']         = opts.n_v
    l_kwargs['n_time']      = opts.n_t
    l_kwargs['t_start']     = opts.init_t
    l_kwargs['t_scale']     = opts.t_scale

    # set intrinsic parameters bounds
    mej_bounds  = [[mmin, mmax] for mmin, mmax in zip(opts.mej_min, opts.meh_max)]
    vel_bounds  = [[vmin, vmax] for vmin, vmax in zip(opts.vel_min, opts.vel_max)]
    opac_bounds = [[omin, omax] for omin, omax in zip(opts.opac_min, opts.opac_max)]


    # define priors
    priors = initialize_knprior(comps=opts.comps, mej_bounds=mej_bounds, vel_bounds=vel_bounds, opac_bounds=opac_bounds, t_gps=opts.t_gps,
                                dist_max=opts.dist_max, dist_min=opts.dist_min,
                                eps0_max=opts.eps_max, eps0_min=opts.eps_min,
                                dist_flag=opts.dist_flag, log_eps0_flag=opts.log_eps_flag,
                                heating_sampling=opts.heat_sampling, heating_alpha=opts.heating_alpha,
                                heating_time=opts.heating_time,heating_sigma=opts.heating_sigma,
                                time_shift_bounds=[opts.time_shift_min, opts.time_shift_max],
                                fixed_names=opts.fixed_names, fixed_values=opts.fixed_values,
                                prior_grid=opts.priorgrid, kind='linear')

    # save observations in pickle
    cont_kwargs = {'filters': l_kwargs['filters']}
    save_container(opts.outdir+'/kn_obs.pkl', cont_kwargs)
    return l_kwargs, priors

def initialize_gwlikelihood_kwargs(opts):

    from ..obs.gw.noise import Noise
    from ..obs.gw.strain import Series
    from ..obs.gw.detector import Detector
    from ..obs.gw.utils import read_data, read_asd, read_spcal

    # initial check
    if len(opts.ifos) != len(opts.strains):
        logger.error("Number of IFOs {} does not match the number of data {}. Please give in input the same number of arguments in the respective order.".format(len(opts.ifos), len(opts.strains)))
        raise ValueError("Number of IFOs {} does not match the number of data {}. Please give in input the same number of arguments in the respective order.".format(len(opts.ifos), len(opts.strains)))
    elif len(opts.ifos) != len(opts.asds):
        logger.error("Number of IFOs does not match the number of ASDs. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of IFOs does not match the number of ASDs. Please give in input the same number of arguments in the respective order.")
    if opts.f_max > opts.srate/2.:
        logger.error("Requested f_max greater than f_Nyquist, outside of the Fourier domain. Please use f_max <= f_Nyq.")
        raise ValueError("Requested f_max greater than f_Nyquist, outside of the Fourier domain. Please use f_max <= f_Nyq.")
    if opts.time_shift_min == None:
        opts.time_shift_min = -opts.time_shift_max

    # initialise dictionaries for detectors, noises, data, etc
    strains   = {}
    dets    = {}
    noises  = {}
    spcals  = {}

    # initialize likelihood keyword arguments
    l_kwargs = {}

    # spcal check
    if len(opts.spcals) == 0 and opts.nspcal > 0:
        logger.warning("Requested number of SpCal nodes > 0 but none SpCal file is given. Ingoring SpCal parameters.")
        opts.nspcal = 0

    # check for PSD weights with frequency binning
    if opts.binning and opts.nweights != 0:
        logger.warning("Requested PSD weights > 0 and frequency binning. These two options are not supported together, PSD-weights are fixed to 0.")
        opts.nweights = 0

    # set up data, detectors and PSDs
    for i,ifo in enumerate(opts.ifos):
        # read data
        ifo         = opts.ifos[i]
        data        = read_data(opts.data_flag , opts.strains[i])
        f_asd , asd = read_asd(opts.asds[i], ifo)

        if opts.binning:
            # if frequency binning is on, the frequency series does not need to be cut
            strains[ifo]      = Series('time' , data ,
                                       srate=opts.srate, seglen=opts.seglen, f_min=opts.f_min, f_max=opts.f_max, t_gps=opts.t_gps,
                                       only=False, alpha_taper=opts.alpha)

        else:
            strains[ifo]      = Series('time' , data ,
                                       srate=opts.srate, seglen=opts.seglen, f_min=opts.f_min, f_max=opts.f_max, t_gps=opts.t_gps,
                                       only=False, alpha_taper=opts.alpha)

        dets[ifo]       = Detector(ifo, t_gps=opts.t_gps)
        noises[ifo]     = Noise(f_asd, asd, f_max=opts.f_max)

        if opts.nspcal > 0:
            spcals[ifo] = read_spcal(opts.spcals[i], ifo)
        else:
            spcals = None

    # check frequency axes
    for ifo1 in opts.ifos:
        for ifo2 in opts.ifos:
            if np.sum((strains[ifo1].freqs != strains[ifo2].freqs)):
                logger.error("Frequency axes for {} data and {} data do not agree.".format(ifo1,ifo2))
                raise ValueError("Frequency axes for {} data and {} data do not agree.".format(ifo1,ifo2))

    freqs   = strains[opts.ifos[0]].freqs

    l_kwargs['ifos']            = opts.ifos
    l_kwargs['datas']           = strains
    l_kwargs['dets']            = dets
    l_kwargs['noises']          = noises
    l_kwargs['freqs']           = freqs
    l_kwargs['srate']           = opts.srate
    l_kwargs['seglen']          = opts.seglen
    l_kwargs['approx']          = opts.approx
    l_kwargs['nspcal']          = opts.nspcal
    l_kwargs['nweights']        = opts.nweights
    l_kwargs['marg_phi_ref']    = opts.marg_phi_ref
    l_kwargs['marg_time_shift'] = opts.marg_time_shift

    # check for extra parameters
    if opts.ej_flag :
        # energy
        if opts.e_min != None and opts.e_max != None:
            e_bounds = [opts.e_min,opts.e_max]
        else:
            e_bounds = None
        # angular momentum
        if opts.j_min != None and opts.j_max != None:
            j_bounds = [opts.j_min,opts.j_max]
        else:
            j_bounds = None
    else:
        e_bounds=None
        j_bounds=None

    # check for extra parameters
    if opts.ecc_flag :
        # eccentricity
        if opts.ecc_min != None and opts.ecc_max != None:
            ecc_bounds=[opts.ecc_min,opts.ecc_max]
        else:
            ecc_bounds = None
    else:
        ecc_bounds=None

    # define priors
    priors, l_kwargs['spcal_freqs'], l_kwargs['len_weights'] = initialize_gwprior(opts.ifos, [opts.mchirp_min,opts.mchirp_max],opts.q_max,
                                                                                              opts.f_min, opts.f_max, opts.t_gps, opts.seglen, opts.srate, opts.approx,
                                                                                              freqs, spin_flag=opts.spin_flag, lambda_flag=opts.lambda_flag,
                                                                                              spin_max=opts.spin_max, lambda_max=opts.lambda_max, lambda_min=opts.lambda_min,
                                                                                              dist_max=opts.dist_max, dist_min=opts.dist_min,
                                                                                              dist_flag=opts.dist_flag,
                                                                                              time_shift_bounds=[opts.time_shift_min, opts.time_shift_max],
                                                                                              fixed_names=opts.fixed_names, fixed_values=opts.fixed_values,
                                                                                              spcals = spcals, nspcal = opts.nspcal , nweights = opts.nweights,
                                                                                              ej_flag = opts.ej_flag, ecc_flag = opts.ecc_flag,
                                                                                              energ_bounds=e_bounds, angmom_bounds=j_bounds, ecc_bounds=ecc_bounds,
                                                                                              marg_phi_ref = opts.marg_phi_ref, marg_time_shift = opts.marg_time_shift,
                                                                                              tukey_alpha = opts.alpha, lmax = opts.lmax,
                                                                                              prior_grid=opts.priorgrid, kind='linear')

    # set fiducial waveform params for binning
    if opts.binning :

        if opts.fiducial == None:
            opts.fiducial = opts.outdir + '/../params.ini'

        # extract parameters for fiducial waveform
        # and fill the dictionary with missing info
        # (like t_gps, f_min and stuff like that)
        from ..obs.gw.utils import read_params
        fiducial_params = read_params(opts.fiducial, flag='fiducial')

        # include spcal env and psd weights in parameter
        # for waveform generation, if needed
        if opts.nspcal > 0 :
            for ni in priors.names:
                if 'spcal' in ni:
                    fiducial_params[ni] = 0.
        if opts.nweights > 0 :
            for ni in priors.names:
                if 'weight' in ni:
                    fiducial_params[ni] = 1.

        fiducial_params =  priors.this_sample(fiducial_params)
        l_kwargs['fiducial_params'] = fiducial_params

    # save observations in pickle
    cont_kwargs = {'datas': strains, 'dets': dets, 'noises': noises}
    save_container(opts.outdir+'/gw_obs.pkl', cont_kwargs)
    return l_kwargs, priors

# auxiliary GW priors

def log_prior_spin_align_volumetric(x, spin_max):
    V   = (4./3.)*np.pi*(spin_max**3.)
    return np.log(0.75*(spin_max*spin_max - x*x))-np.log(np.abs(V))

def log_prior_spin_align_isotropic(x, spin_max):
    logp = np.log(-np.log(np.abs(x/spin_max)) ) - np.log(2.0 * np.abs(spin_max))
    if np.isinf(logp):
        return -np.inf
    else:
        return logp

def log_prior_spin_precess_volumetric(x , spin_max):
    V   = (spin_max**3.)/3.
    return 2.*np.log(x) - np.log(np.abs(V))

def log_prior_spin_precess_isotropic(x, spin_max):
    return -np.log(np.abs(spin_max))

def log_prior_massratio(x, q_max):
    from scipy.special import hyp2f1
    n  = 5.*(hyp2f1(-0.4, -0.2, 0.8, -1.)-hyp2f1(-0.4, -0.2, 0.8, -q_max)/(q_max**0.2))
    return 0.4*np.log((1.+x)/(x**3.))-np.log(np.abs(n))

def log_prior_comoving_volume(x, cosmo):
    dvc_ddl   = cosmo.dvc_ddl(x)
    return np.log(dvc_ddl)

def log_prior_sourceframe_volume(x, cosmo):
    dvc_ddl = cosmo.dvc_ddl(x)
    z       = cosmo.dl_to_z(x)
    return np.log(dvc_ddl) - np.log(1.+z)

# prior helpers

def fill_params_from_dict(dict):
    """
        Return list of names and bounds and dictionary of constants properties
        """
    from ..inf.prior import Parameter, Variable, Constant
    params  = []
    variab  = []
    const   = []
    for k in dict.keys():
        if isinstance(dict[k], Parameter):
            params.append(dict[k])
        elif isinstance(dict[k], Variable):
            variab.append(dict[k])
        elif isinstance(dict[k], Constant):
            const.append(dict[k])
    return  params, variab, const

def initialize_gwprior(ifos, mchirp_bounds, q_max, f_min, f_max, t_gps, seglen, srate, approx, freqs,
                       spin_flag='no-spins', lambda_flag='no-tides',
                       spin_max=None, lambda_max=None, lambda_min=None,
                       dist_max=None, dist_min=None, dist_flag='vol',
                       time_shift_bounds=None,
                       fixed_names=[], fixed_values=[],
                       spcals=None, nspcal=0, nweights=0,
                       ej_flag = False, ecc_flag = False,
                       energ_bounds=None, angmom_bounds=None, ecc_bounds=None,
                       marg_phi_ref=False, marg_time_shift=False,
                       tukey_alpha=None, lmax=2,
                       prior_grid=2000, kind='linear'):

    from ..inf.prior import Prior, Parameter, Variable, Constant

    names   = []
    bounds  = []
    funcs   = []
    kwargs  = {}

    interp_kwarg = {'ngrid': prior_grid, 'kind': kind}

    # avoid mchirp,q in fixed names
    if 'mchirp' in fixed_names or 'q' in fixed_names:
        logger.error("Unable to set masses as constant properties. The prior does not support this function yet.")
        raise RuntimeError("Unable to set masses as constant properties. The prior does not support this function yet.")

    # wrap everything into a dictionary
    dict = {}

    # setting masses (mchirp,q)
    # from scipy.special import hyp2f1
    # norm_q  = 5.*(hyp2f1(-0.4, -0.2, 0.8, -1.)-hyp2f1(-0.4, -0.2, 0.8, -q_max)/np.power(q_max, 0.2))

    dict['mchirp']  = Parameter(name='mchirp',
                                min=mchirp_bounds[0],
                                max=mchirp_bounds[1],
                                prior='linear')

    dict['q']       = Parameter(name='q',
                                min=1.,
                                max=q_max,
                                func=log_prior_massratio,
                                func_kwarg={'q_max': q_max},
                                interp_kwarg=interp_kwarg)

    # setting spins
    if spin_max != None:
        if spin_max > 1.:
            logger.warning("Input spin-max is greater than 1, this is not a physical value. The input value will be ignored and spin-max will be 1.")
            spin_max = 1.

    if spin_flag == 'no-spins':
        dict['s1x'] = Constant('s1x', 0.)
        dict['s2x'] = Constant('s2x', 0.)
        dict['s1y'] = Constant('s1y', 0.)
        dict['s2y'] = Constant('s2y', 0.)
        dict['s1z'] = Constant('s1z', 0.)
        dict['s2z'] = Constant('s2z', 0.)

    else:

        if spin_max == None:
            logger.error("Spinning model requested without input maximum spin specification. Please include argument spin_max in Prior")
            raise ValueError("Spinning model requested without input maximum spin specification. Please include argument spin_max in Prior")

        elif spin_flag == 'align-volumetric':

            dict['s1z'] = Parameter(name='s1z',
                                    min=-spin_max,
                                    max=spin_max,
                                    func=log_prior_spin_align_volumetric,
                                    func_kwarg={'spin_max':spin_max},
                                    interp_kwarg=interp_kwarg)

            dict['s2z'] = Parameter(name='s2z',
                                    min=-spin_max,
                                    max=spin_max,
                                    func=log_prior_spin_align_volumetric,
                                    func_kwarg={'spin_max':spin_max},
                                    interp_kwarg=interp_kwarg)

            dict['s1x'] = Constant('s1x', 0.)
            dict['s2x'] = Constant('s2x', 0.)
            dict['s1y'] = Constant('s1y', 0.)
            dict['s2y'] = Constant('s2y', 0.)

        elif spin_flag == 'align-isotropic':

            dict['s1z'] = Parameter(name='s1z',
                                    min=-spin_max,
                                    max=spin_max,
                                    func=log_prior_spin_align_isotropic,
                                    func_kwarg={'spin_max':spin_max},
                                    interp_kwarg=interp_kwarg)

            dict['s2z'] = Parameter(name='s2z',
                                    min=-spin_max,
                                    max=spin_max,
                                    func=log_prior_spin_align_isotropic,
                                    func_kwarg={'spin_max':spin_max},
                                    interp_kwarg=interp_kwarg)

            dict['s1x'] = Constant('s1x', 0.)
            dict['s2x'] = Constant('s2x', 0.)
            dict['s1y'] = Constant('s1y', 0.)
            dict['s2y'] = Constant('s2y', 0.)

        elif spin_flag == 'precess-volumetric':
            # if precessing, use polar coordinates for the sampling,
            # the waveform will tranform these values also in cartesian coordinates
            dict['s1']      = Parameter(name='s1',
                                        min=0.,
                                        max=spin_max,
                                        prior='quadratic')
            dict['s2']      = Parameter(name='s2',
                                        min=0.,
                                        max=spin_max,
                                        prior='quadratic')

            dict['tilt1']   = Parameter(name='tilt1',
                                        min=0.,
                                        max=np.pi,
                                        prior='sinusoidal')
            dict['tilt2']   = Parameter(name='tilt2',
                                        min=0.,
                                        max=np.pi,
                                        prior='sinusoidal')

            dict['phi_1l']  = Parameter(name='phi_1l',
                                        min=0.,
                                        max=2.*np.pi,
                                        periodic=1,
                                        prior='uniform')
            dict['phi_2l']  = Parameter(name='phi_2l',
                                        min=0.,
                                        max=2.*np.pi,
                                        periodic=1,
                                        prior='uniform')

        elif spin_flag == 'precess-isotropic':
            # if precessing, use polar coordinates for the sampling,
            # the waveform will tranform these values also in cartesian coordinates
            dict['s1']      = Parameter(name='s1',
                                        min=0.,
                                        max=spin_max,
                                        prior_func='uniform')
            dict['s2']      = Parameter(name='s2',
                                        min=0.,
                                        max=spin_max,
                                        prior_func='uniform')

            dict['tilt1']   = Parameter(name='tilt1',
                                        min=0.,
                                        max=np.pi,
                                        prior='sinusoidal')
            dict['tilt2']   = Parameter(name='tilt2',
                                        min=0.,
                                        max=np.pi,
                                        prior='sinusoidal')

            dict['phi_1l']  = Parameter(name='phi_1l',
                                        min=0.,
                                        max=2.*np.pi,
                                        periodic=1,
                                        prior='uniform')
            dict['phi_2l']  = Parameter(name='phi_2l',
                                        min=0.,
                                        max=2.*np.pi,
                                        periodic=1,
                                        prior='uniform')

        else:
            logger.error("Unable to read spin flag for Prior. Please use one of the following: 'no-spins', 'align-isotropic', 'align-volumetric', 'precess-isotropic', 'precess-volumetric'")
            raise ValueError("Unable to read spin flag for Prior. Please use one of the following: 'no-spins', 'align-isotropic', 'align-volumetric', 'precess-isotropic', 'precess-volumetric'")

    # setting lambdas
    if lambda_flag == 'no-tides':
        dict['lambda1'] = Constant('lambda1', 0.)
        dict['lambda2'] = Constant('lambda2', 0.)

    else:

        if lambda_min == None:
            lambda_min = 0.

        if lambda_max == None:
            logger.error("Tidal model requested without input maximum lambda specification. Please include argument lambda_max in Prior")
            raise ValueError("Tidal model requested without input maximum lambda specification. Please include argument lambda_max in Prior")

        if lambda_flag == 'bns-tides':
            dict['lambda1'] = Parameter(name='lambda1',
                                        min=lambda_min,
                                        max=lambda_max,
                                        prior='uniform')
            dict['lambda2'] = Parameter(name='lambda2',
                                        min=lambda_min,
                                        max=lambda_max,
                                        prior='uniform')

        elif lambda_flag == 'bns-eos4p':
            dict['eos_gamma0'] = Parameter(name='eos_gamma0', min=32., max=35.)
            dict['eos_gamma1'] = Parameter(name='eos_gamma1', min=1.01, max=5.)
            dict['eos_gamma2'] = Parameter(name='eos_gamma2', min=1.01, max=5.)
            dict['eos_gamma3'] = Parameter(name='eos_gamma3', min=1.01, max=5.)

        elif lambda_flag == 'bhns-tides':
            dict['lambda1'] = Constant('lambda1', 0.)
            dict['lambda2'] = Parameter(name='lambda2',
                                        min=lambda_min,
                                        max=lambda_max,
                                        prior='uniform')

        elif lambda_flag == 'bhns-eos4p':
            dict['lambda1'] = Constant('lambda1', 0.)
            dict['eos_gamma0'] = Parameter(name='eos_gamma0', min=32., max=35.)
            dict['eos_gamma1'] = Parameter(name='eos_gamma1', min=1.01, max=5.)
            dict['eos_gamma2'] = Parameter(name='eos_gamma2', min=1.01, max=5.)
            dict['eos_gamma3'] = Parameter(name='eos_gamma3', min=1.01, max=5.)

        elif lambda_flag == 'nsbh-tides':
            dict['lambda1'] = Parameter(name='lambda1',
                                        min=lambda_min,
                                        max=lambda_max,
                                        prior='uniform')
            dict['lambda2'] = Constant('lambda2', 0.)

        elif lambda_flag == 'nsbh-eos4p':
            dict['lambda2'] = Constant('lambda2', 0.)
            dict['eos_gamma0'] = Parameter(name='eos_gamma0', min=32., max=35.)
            dict['eos_gamma1'] = Parameter(name='eos_gamma1', min=1.01, max=5.)
            dict['eos_gamma2'] = Parameter(name='eos_gamma2', min=1.01, max=5.)
            dict['eos_gamma3'] = Parameter(name='eos_gamma3', min=1.01, max=5.)

        else:
            logger.error("Unable to read tidal flag for Prior. Please use one of the following: 'no-tides', 'bns-tides', 'bhns-tides', 'nsbh-tides' or flags for parametrized EOS.")
            raise ValueError("Unable to read tidal flag for Prior. Please use one of the following: 'no-tides', 'bns-tides', 'bhns-tides', 'nsbh-tides' or flags for parametrized EOS.")

    # setting sky position
    dict['ra']  = Parameter(name='ra', min=0., max=2.*np.pi, periodic=1)
    dict['dec'] = Parameter(name='dec', min=-np.pi/2., max=np.pi/2., prior='cosinusoidal')

    # setting other extrinsic parameters
    dict['cosi']    = Parameter(name='cosi', min=-1., max=+1.)
    dict['psi']     = Parameter(name='psi', min=0., max=np.pi, periodic=1)

    # setting distance
    if dist_min == None and dist_max == None:
        logger.warning("Requested bounds for distance parameter is empty. Setting standard bound [10,1000] Mpc")
        dist_min = 10.
        dist_max = 1000.
    elif dist_min == None:
        logger.warning("Requested lower bound for distance parameter is empty. Setting standard bound 10 Mpc")
        dist_min = 10.
    elif dist_max == None:
        logger.warning("Requested upper bound for distance parameter is empty. Setting standard bound 1. Gpc")
        dist_max = 1000.

    if dist_flag=='log':
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='log-uniform')
    elif dist_flag=='vol':
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='quadratic')
    elif dist_flag=='com':
        from ..obs.utils.cosmo import Cosmology
        cosmo = Cosmology(cosmo='Planck18_arXiv_v2', kwargs=None)
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       func=log_prior_comoving_volume,
                                       func_kwarg={'cosmo': cosmo},
                                       interp_kwarg=interp_kwarg)
    elif dist_flag=='src':
        from ..obs.utils.cosmo import Cosmology
        cosmo = Cosmology(cosmo='Planck18_arXiv_v2', kwargs=None)
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       func=log_prior_sourceframe_volume,
                                       func_kwarg={'cosmo': cosmo},
                                       interp_kwarg=interp_kwarg)
    else:
        logger.error("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")
        raise RuntimeError("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")

    # setting time_shift
    if marg_time_shift:
        dict['time_shift'] = Constant('time_shift',0.)
    else:
        if time_shift_bounds == None:
            logger.warning("Requested bounds for time_shift parameter is empty. Setting standard bound [-1.0,+1.0] s")
            time_shift_bounds = [-1.0,+1.]

        dict['time_shift'] = Parameter(name='time_shift', min=time_shift_bounds[0], max=time_shift_bounds[1])

    # setting phi_ref
    if marg_phi_ref:
        dict['phi_ref']    = Constant('phi_ref',0.)
    else:
        dict['phi_ref']    = Parameter(name='phi_ref',
                                       min=0.,
                                       max=2.*np.pi,
                                       periodic=1)

    # include PSD weights, if requested
    if nweights != 0:

        f_cut       = np.logspace(1., np.log(np.max(freqs))/np.log(np.min(freqs)), base=np.min(freqs), num = nweights+1)
        len_weights = np.array([len(freqs[np.where((freqs>=f_cut[i])&(freqs<f_cut[i+1]))]) for i in range(nweights)])
        len_weights[-1] += 1

        if np.sum(len_weights) != len(freqs):

            if np.sum(len_weights) > len(freqs):
                dn = int(np.sum(len_weights)) - len(freqs)
                len_weights[-1] -= dn
            elif np.sum(len_weights) < len(freqs):
                dn = len(freqs) - int(np.sum(len_weights))
                len_weights[-1] += dn

        for i in range(nweights):
            for ifo in ifos:
                sigma_w = 1./np.sqrt(len_weights[i])
                # bounds centered on 1 with width of 5 sigma
                dict['weight{}_{}'.format(i,ifo)] = Parameter(name='weight{}_{}'.format(i,ifo),
                                                              min = np.max([0, 1.-5.*sigma]),
                                                              max = 1.+5.*sigma,
                                                              prior='normal',
                                                              mu = 1., sigma = sigma_w)
    else:
        len_weights = None

    # include SpCal envelopes, if requested
    if nspcal != 0:

        if spcals == None:
            logger.error("Impossible to determine calibration prior. SpCal files are missing.")
            raise ValueError("Impossible to determine calibration prior. SpCal files are missing.")

        if nspcal < 2:
            logger.warning("Impossible to use only one SpCal node. Setting 2 nodes.")
            nspcal = 2

        freqs          = freqs
        spcal_freqs    = np.logspace(1., np.log(np.max(freqs))/np.log(np.min(freqs)), base=np.min(freqs), num = nspcal)

        spcal_amp_sigmas = {}
        spcal_phi_sigmas = {}

        for ifo in ifos:
            spcal_amp_sigmas[ifo] =np.interp(spcal_freqs,spcals[ifo][0],spcals[ifo][1])
            spcal_phi_sigmas[ifo] =np.interp(spcal_freqs,spcals[ifo][0],spcals[ifo][2])

        for i in range(nspcal):
            for ifo in ifos:

                dict['spcal_amp{}_{}'.format(i,ifo)] = Parameter(name='spcal_amp{}_{}'.format(i,ifo),
                                                                 min = -5.*spcal_amp_sigmas[ifo][i],
                                                                 max = 5.*spcal_amp_sigmas[ifo][i],
                                                                 prior='normal', mu = 0.,
                                                                 sigma = spcal_amp_sigmas[ifo][i])


                dict['spcal_phi{}_{}'.format(i,ifo)] = Parameter(name='spcal_phi{}_{}'.format(i,ifo),
                                                                 max = 5.*spcal_phi_sigmas[ifo][i],
                                                                 min = -5.*spcal_phi_sigmas[ifo][i],
                                                                 prior='normal', mu = 0.,
                                                                 sigma = spcal_phi_sigmas[ifo][i])

    else:
        spcal_freqs = None

    # include extra parameters: energy and angular momentum
    if ej_flag:

        if energ_bounds == None:
            logger.warning("Requested bounds for energy parameter is empty. Setting standard bound [0.95,1.5]")
            energ_bounds = [1.0001,1.1]

        dict['energy'] = Parameter(name='energy', min=energ_bounds[0], max=energ_bounds[1])

        if angmom_bounds == None:
            logger.warning("Requested bounds for angular momentum parameter is empty. Setting standard bound [3,5]")
            angmom_bounds = [3.5,4.5]

        dict['angmom'] = Parameter(name='angmom', min=angmom_bounds[0], max=angmom_bounds[1])

    # include extra parameters: eccentricity
    if ecc_flag:

        if ecc_bounds == None:
            logger.warning("Requested bounds for eccentricity parameter is empty. Setting standard bound [1e-3,1]")
            ecc_bounds = [0.001, 1.]

        dict['eccentricity'] = Parameter(name='eccentricity', min=ecc_bounds[0], max=ecc_bounds[1])

    else:
        dict['eccentricity'] = Constant('eccentricity', 0.)

    # set fixed parameters
    if len(fixed_names) != 0 :
        assert len(fixed_names) == len(fixed_values)
        for ni,vi in zip(fixed_names,fixed_values) :
            if ni not in list(dict.keys()):
                logger.warning("Requested fixed parameters ({}={}) is not in the list of all parameters. The command will be ignored.".format(ni,vi))
                continue
            else:
                dict[ni] = Constant(ni, vi)

    # fill values for the waveform and the likelihood
    dict['f_min']  = Constant('f_min',  f_min)
    dict['f_max']  = Constant('f_max',  f_max)
    dict['t_gps']  = Constant('t_gps',  t_gps)
    dict['seglen'] = Constant('seglen', seglen)
    dict['srate']  = Constant('srate',  srate)
    dict['lmax']   = Constant('lmax',   lmax)

    if tukey_alpha == None:
        tukey_alpha = 0.4/seglen
    dict['tukey']  = Constant('tukey',  tukey_alpha)

    params, variab, const = fill_params_from_dict(dict)

    logger.info("Setting parameters for sampling ...")
    for pi in params:
        logger.info(" - {} in range [{:.2f},{:.2f}]".format(pi.name , pi.bound[0], pi.bound[1]))

    logger.info("Setting constant properties ...")
    for ci in const:
        logger.info(" - {} fixed to {}".format(ci.name , ci.value))

    logger.info("Initializing prior ...")

    return Prior(parameters=params, variables=variab, constants=const), spcal_freqs, len_weights

def initialize_knprior(comps, mej_bounds, vel_bounds, opac_bounds, t_gps,
                       dist_max=None, dist_min=None,
                       eps0_max=None, eps0_min=None,
                       dist_flag=False, log_eps0_flag=False,
                       heating_sampling=False, heating_alpha=1.3, heating_time=1.3, heating_sigma=0.11,
                       time_shift_bounds=None,
                       fixed_names=[], fixed_values=[],
                       prior_grid=2000, kind='linear'):

    from ..inf.prior import Prior, Parameter, Variable, Constant

    # initializing disctionary for wrap up all information
    dict = {}

    # checking number of components and number of prior bounds
    if len(comps) != len(mej_bounds):
        logger.error("Number of Mej bounds does not match the number of components")
        raise ValueError("Number of Mej bounds does not match the number of components")
    if len(comps) != len(vel_bounds):
        logger.error("Number of velocity bounds does not match the number of components")
        raise ValueError("Number of velocity bounds does not match the number of components")
    if len(comps) != len(opac_bounds):
        logger.error("Number of opacity bounds does not match the number of components")
        raise ValueError("Number of opacity bounds does not match the number of components")

    # setting ejecta properties for every component
    for i,ci in enumerate(comps):
        dict['mej_{}'.format(ci)]  = Parameter(name='mej_{}'.format(ci), min = mej_bounds[i][0], max = mej_bounds[i][1])
        dict['vel_{}'.format(ci)]  = Parameter(name='vel_{}'.format(ci), min = vel_bounds[i][0], max = vel_bounds[i][1])
        dict['opac_{}'.format(ci)] = Parameter(name='opac_{}'.format(ci), min = opac_bounds[i][0], max = opac_bounds[i][1])

    # setting eps0
    if eps0_min == None and eps0_max == None:
        logger.warning("Requested bounds for heating parameter eps0 is empty. Setting standard bound [1e17,1e19].")
        eps0_min = 1.e17
        eps0_max = 5.e19
    elif eps0_min == None and eps0_max != None:
        eps0_min = 1.e17
        eps0_max = eps0_max

    if log_eps0_flag:
        dict['eps0']   = Parameter(name='eps0', min = eps0_min, max = eps0_max, prior = 'log-uniform')
    else:
        dict['eps0']   = Parameter(name='eps0', min = eps0_min, max = eps0_max)

    # set heating coefficients
    if heating_sampling:
        logger.warning("Including extra heating coefficiets in sampling using default bounds with uniform prior.")
        dict['eps_alpha']   = Parameter(name='eps_alpha',    min=1., max=10.)
        dict['eps_time']    = Parameter(name='eps_time',     min=0., max=25.)
        dict['eps_sigma']   = Parameter(name='eps_sigma',    min=1.e-5, max=50.)
    else:
        dict['eps_alpha']   = Constant('eps_alpha', heating_alpha)
        dict['eps_time']    = Constant('eps_time',  heating_time)
        dict['eps_sigma']   = Constant('eps_sigma', heating_sigma)

    # setting distance
    if dist_min == None and dist_max == None:
        logger.warning("Requested bounds for distance parameter is empty. Setting standard bound [10,1000] Mpc")
        dist_min = 10.
        dist_max = 1000.
    elif dist_min == None:
        logger.warning("Requested lower bounds for distance parameter is empty. Setting standard bound 10 Mpc")
        dist_min = 10.

    elif dist_max == None:
        logger.warning("Requested bounds for distance parameter is empty. Setting standard bound 1 Gpc")
        dist_max = 1000.

    if dist_flag=='log':
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='log-uniform')
    elif dist_flag=='vol':
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='quadratic')
    elif dist_flag=='com':
        from ..obs.utils.cosmo import Cosmology
        cosmo = Cosmology(cosmo='Planck18_arXiv_v2', kwargs=None)
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       func=log_prior_comoving_volume,
                                       func_kwarg={'cosmo': cosmo},
                                       interp_kwarg=interp_kwarg)
    elif dist_flag=='src':
        from ..obs.utils.cosmo import Cosmology
        cosmo = Cosmology(cosmo='Planck18_arXiv_v2', kwargs=None)
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       func=log_prior_sourceframe_volume,
                                       func_kwarg={'cosmo': cosmo},
                                       interp_kwarg=interp_kwarg)
    else:
        logger.error("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")
        raise RuntimeError("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")

    # setting time_shift
    if time_shift_bounds == None:
        logger.warning("Requested bounds for time_shift parameter is empty. Setting standard bound [-1.0,+1.0] day")
        time_shift_bounds  = [-86400.,+86400.]

    dict['time_shift']  = Parameter(name='time_shift', min=time_shift_bounds[0], max=time_shift_bounds[1])

    # setting inclination
    dict['cosi']   =  Parameter(name='cosi', min=-1., max=+1.)

    # set fixed parameters
    if len(fixed_names) != 0 :
        assert len(fixed_names) == len(fixed_values)
        for ni,vi in zip(fixed_names,fixed_values) :
            if ni not in list(dict.keys()):
                logger.warning("Requested fixed parameters ({}={}) is not in the list of all parameters. The command will be ignored.".format(ni,vi))
            else:
                dict[ni] = Constant(ni, vi)

    dict['t_gps']  = Constant('t_gps', t_gps)

    params, variab, const = fill_params_from_dict(dict)

    logger.info("Setting parameters for sampling ...")
    for pi in params:
        logger.info(" - {} in range [{:.2f},{:.2f}]".format(pi.name , pi.bound[0], pi.bound[1]))

    logger.info("Setting constant properties ...")
    for ci in const:
        logger.info(" - {} fixed to {}".format(ci.name , ci.value))

    logger.info("Initializing prior ...")

    return Prior(parameters=params, variables=variab, constants=const)


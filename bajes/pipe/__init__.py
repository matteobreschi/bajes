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
    import h5py
    utf8_type = h5py.string_dtype('utf-8', 30)
except ImportError:
    pass

# logger

import logging
logger = logging.getLogger(__name__)

# imports from pipe modules

from .utils import is_picklable, save_container, data_container
from .gw_init import initialize_gwlikelihood_kwargs
from .kn_init import initialize_knlikelihood_kwargs

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

# hdf5

def save_dict_to_hdf5(dic, path, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, path, dic)

def _try_convertable_list(key, item):
    try:
        _item = np.array(item,dtype=np.float64)
    except Exception:
        try:
            _item =np.array(item,dtype=utf8_type)
        except Exception:
            raise ValueError('Cannot save {} object of {} type'.format(key,type(item)))
    return _item

def _try_convertable_item(key, item):
    try:
        _item = np.float64(item)
    except Exception:
        try:
            _item = str(item)
        except Exception:
            raise ValueError('Cannot save {} object of {} type'.format(key,type(item)))
    return _item

def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, (np.ndarray, list)):
            h5file[path + key] = _try_convertable_list(key, item)
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            h5file[path + key] = _try_convertable_item(key, item)

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

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

# pipeline/core methods

def parse_main_options():

    from .. import __version__, __doc__
    from ..inf import __known_samplers__
    import optparse as op

    usage   = "python -m bajes [options]\n"+"Version: bajes {}".format(__version__)
    parser=op.OptionParser(usage=usage, version=__version__, description="Description:\n"+__doc__)

    parser.add_option('-p', '--prior',  dest='prior',       type='string',  help='path to prior file (configuration file)')
    parser.add_option('-l', '--like',   dest='like',        type='string',  help='path to likelihood function (python file)')

    # Choose the engine
    parser.add_option('-s', '--sampler', dest='engine',      default='dynesty', type='string',  help='sampler engine name, {}'.format(__known_samplers__))

    # Parallelization option (only multiprocessing)
    parser.add_option('-n', '--nprocs',       dest='nprocs',    default=None,   type='int', help='number of parallel processes')

    # output
    parser.add_option('-o', '--outdir', default='./',       type='string',  dest='outdir',  help='output directory')

    # Nested sampling options
    parser.add_option('--nlive',        dest='nlive',       default=1024,   type='int',     help='[nest] number of live points. Default: 1024')
    parser.add_option('--tol',          dest='tolerance',   default=0.1,    type='float',   help='[nest] evidence tolerance. Default: 0.1')
    parser.add_option('--maxmcmc',      dest='maxmcmc',     default=4096,   type='int',     help='[nest] maximum number of mcmc iterations. Default: 4096')
    parser.add_option('--minmcmc',      dest='minmcmc',     default=32,     type='int',     help='[nest] minimum number of mcmc iterations. Default: 32')
    parser.add_option('--poolsize',     dest='poolsize',    default=2048,   type='int',     help='[nest] number of sample in the pool (cpnest). Default: 2048')
    parser.add_option('--nact',         dest='nact',        default=5,      type='int',     help='[nest] sub-chain safe factor (dynesty). Default: 5')
    parser.add_option('--nbatch',       dest='nbatch',      default=512,    type='int',     help='[nest] number of live points for batch (dynesty-dyn). Default: 512')
    parser.add_option('--dkl',          dest='dkl',         default=0.5,    type='float',   help='[nest] target KL divergence (ultranest). Default: 0.5')
    parser.add_option('--z-frac',       dest='z_frac',      default=None,   type='float',   help='[nest] remaining Z fraction (ultranest). Default: None')

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

    # Generic run options
    parser.add_option('--tag',               dest='tags',           default=[],        type='string', action="append",       help="Tag for data messenger. Available options: ['gw', 'kn'].")
    parser.add_option('--t-gps',             dest='t_gps',                             type='float',                         help='GPS time: for GW, center value of time axis (if a local version of the data is provided, has to coincide with this value for the input time axis); for KN, initial value of time axis')
    parser.add_option('--engine',            dest='engine',         default='dynesty', type='string',                        help='Sampler engine name, available options: {}.'.format(__known_samplers__))
    parser.add_option('--priorgrid',         dest='priorgrid',      default=1000,      type='int',                           help='number of nodes for prior interpolators (if needed)')

    # I/O options
    parser.add_option('-o', '--outdir',      dest='outdir',         default=None,      type='string',                        help='directory for output')
    parser.add_option('--debug',             dest='debug',          default=False,                     action="store_true",  help='use debugging mode for logger')
    parser.add_option('--verbose',           dest='silence',        default=True,                      action="store_false", help='activate stream handler, use this if you are running on terminal')
    parser.add_option('--tracing',           dest='trace_memory',   default=False,                     action="store_true",  help='keep track of memory usage')

    # Nested sampling options
    parser.add_option('--nlive',             dest='nlive',          default=1024,      type='int',                           help='number of live points')
    parser.add_option('--tol',               dest='tolerance',      default=0.1,       type='float',                         help='evidence tolerance')
    parser.add_option('--maxmcmc',           dest='maxmcmc',        default=4096,      type='int',                           help='maximum number of mcmc iterations')
    parser.add_option('--minmcmc',           dest='minmcmc',        default=32,        type='int',                           help='minimum number of mcmc iterations')
    parser.add_option('--poolsize',          dest='poolsize',       default=2048,      type='int',                           help='number of sample in the pool (cpnest)')
    parser.add_option('--nact',              dest='nact',           default=5,         type='int',                           help='sub-chain safe factor (dynesty)')
    parser.add_option('--nbatch',            dest='nbatch',         default=512,       type='int',                           help='number of live points for batch (dynesty-dyn)')
    parser.add_option('--dkl',               dest='dkl',            default=0.5,       type='float',                         help='target KL divergence (ultranest)')
    parser.add_option('--z-frac',            dest='z_frac',         default=None,      type='float',                         help='remaining Z fraction (ultranest)')

    # MCMC options
    parser.add_option('--nout',              dest='nout',           default=4000,      type='int',                           help='number of posterior samples')
    parser.add_option('--nwalk',             dest='nwalk',          default=256,       type='int',                           help='number of parallel walkers')
    parser.add_option('--nburn',             dest='nburn',          default=15000,     type='int',                           help='numebr of burn-in iterations')
    parser.add_option('--ntemp',             dest='ntemps',         default=8,         type='int',                           help='number of tempered ensambles (ptmcmc)')
    parser.add_option('--tmax',              dest='tmax',           default=None,      type='float',                         help='maximum temperature scale, default inf (ptmcmc)')

    # Parallelization options
    parser.add_option('--nprocs',            dest='nprocs',         default=None,      type='int',                           help='number of processes in the pool')
    parser.add_option('--mpi-per-node',      dest='mpi_per_node',   default=None,      type='int',                           help='number of MPI processes per node')
    parser.add_option('--fast-mpi',          dest='fast_mpi',       default=False,                     action="store_true",  help='enable fast MPI communication')

    # Other samplers options
    parser.add_option('--use-slice',         dest='use_slice',      default=False,                     action="store_true",  help='use slice proposal (emcee or cpnest)')
    parser.add_option('--checkpoint',        dest='ncheck',         default=0,         type='int',                           help='number of periodic checkpoints')
    parser.add_option('--seed',              dest='seed',           default=None,      type='int',                           help='seed for the pseudo-random chain')

    # Fixed parameter options
    parser.add_option('--fix-name',          dest='fixed_names',    default=[],        type='string',  action="append",      help='names of fixed params')
    parser.add_option('--fix-value',         dest='fixed_values',   default=[],        type='float',   action="append",      help='values of fixed params')

    # Distance information
    parser.add_option('--dist-flag',         dest='dist_flag',      default='vol',     type='string',                        help='distance prior flag (options: vol, log, com, src)')
    parser.add_option('--dist-min',          dest='dist_min',       default=[],        type='float',   action="append",      help='lower distance prior bound')
    parser.add_option('--dist-max',          dest='dist_max',       default=[],        type='float',   action="append",      help='upper distance prior bound')

    # Time shift (from GPS time) information
    parser.add_option('--tshift-max',        dest='time_shift_max', default=[],        type='float',   action="append",      help='upper time shift prior bound')
    parser.add_option('--tshift-min',        dest='time_shift_min', default=[],        type='float',   action="append",      help='lower time shift prior bound')

    #
    # GW OPTIONS
    #

    # Data and PSDs information
    parser.add_option('--ifo',               dest='ifos',           default=[],        type='string',  action="append",      help='Detector to be considered in the analysis. Has to be passed once per detector, and sets the order for similar commands to pass strains and psds. Available options: [H1, L1, V1, K1, G1]. Default: []')
    parser.add_option('--strain',            dest='strains',        default=[],        type='string',  action="append",      help='Path to strain data. Has to be passed once per detector and in the same order as the `--ifo` options. Default: []')
    parser.add_option('--asd',               dest='asds',           default=[],        type='string',  action="append",      help='Path to ASD data. Has to be passed once per detector and in the same order as the `--ifo` options. Default: []')
    parser.add_option('--alpha',             dest='alpha',          default=None,      type='float',                         help='Alpha parameter of the Tukey window. Default: 0.4/seglen.')

    # Optional, calibration envelopes
    parser.add_option('--spcal',             dest='spcals',         default=[],        type='string',  action="append",      help='Path to calibration envelope. Has to be passed once per detector and in the same order as the `--ifo` options. Default: []')
    parser.add_option('--nspcal',            dest='nspcal',         default=0,         type='int',                           help='Number of spectral calibration nodes. Default: 0')

    # Time series information
    parser.add_option('--f-min',             dest='f_min',          default=None,      type='float',                         help='Mandatory parameter: minimum frequency at which likelihood is evaluated [Hz]. Default: None')
    parser.add_option('--f-max',             dest='f_max',          default=None,      type='float',                         help='Mandatory parameter: maximum frequency at which likelihood is evaluated [Hz]. Default: None')
    parser.add_option('--srate',             dest='srate',          default=None,      type='float',                         help='Mandatory parameter: requested sampling rate [Hz]. If smaller than data samppling rate, downsampling is applied. Default: None')
    parser.add_option('--seglen',            dest='seglen',         default=None,      type='float',                         help='Mandatory parameter: requested length of the segment to be analysed [sec]. If smaller than data total lenght, data are cropped. If longer, data are padded. Default: None.')

    # Waveform model
    parser.add_option('--approx',            dest='approx',         default=None,      type='string',                        help='Gravitational-wave approximant. Default: None')
    parser.add_option('--extra-option',      dest='extra_opt',      default=[],        type='string',  action="append",      help='Names of the additional parameters for the chosen approximant. Has to be passed once for each parameter. Default: []')
    parser.add_option('--extra-option-val',  dest='extra_opt_val',  default=[],        type='string',  action="append",      help='Values of the additional parameters for the chosen approximant. Has to be passed once for each parameter and in the same order as the `--extra-option` option. Default: []')
    parser.add_option('--lmax',              dest='lmax',           default=0,         type='int',                           help='Higher angular mode index to be considered for GW template. Default: 0')
    parser.add_option('--Eprior',            dest='Eprior',         default=None,      type='str',                           help='Prior on the initial energy in hyperbolic TEOB models. Default: None. Available options: [Constrained, Unconstrained].')
    parser.add_option('--nqc-TEOBHyp',       dest='nqc_TEOBHyp',    default=1,         type='int',                           help='Flag to activate/deactivate (1/0) NQCs in hyperbolic TEOB models. Default: 1.')

    # Prior flags
    parser.add_option('--data-flag',         dest='data_flag',      default=None,       type='string',                       help='Data flag. Available options: [???]. Default: None')
    parser.add_option('--spin-flag',         dest='spin_flag',      default='no-spins', type='string',                       help='Spin prior flag. Available options: [???]. Default: `no-spins`.')
    parser.add_option('--tidal-flag',        dest='lambda_flag',    default='no-tides', type='string',                       help='Tidal prior flag. Available options: [???]. Default: `no-tides`.')

    # Prior bounds
    parser.add_option('--mc-min',            dest='mchirp_min',      default=None,   type='float',   help='lower mchirp prior bound (if use-mtot, lower mtot prior bound)')
    parser.add_option('--mc-max',            dest='mchirp_max',      default=None,   type='float',   help='upper mchirp prior bound (if use-mtot, upper mtot prior bound)')
    parser.add_option('--q-max',             dest='q_max',           default=None,   type='float',   help='upper mass ratio prior bound')
    parser.add_option('--q-min',             dest='q_min',           default=1.,     type='float',   help='lower mass ratio prior bound')
    #parser.add_option('--mass-max',        dest='mass_max',       default=None,   type='float',   help='upper mass component prior bound')
    #parser.add_option('--mass-min',        dest='mass_min',       default=None,   type='float',   help='lower mass component prior bound')
    parser.add_option('--spin-max',          dest='spin_max',        default=None,   type='float',   help='upper spin prior bound')
    parser.add_option('--spin1-max',         dest='spin1_max',      default=None,   type='float',   help='upper spin prior bound')
    parser.add_option('--spin2-max',         dest='spin2_max',      default=None,   type='float',   help='upper spin prior bound')
    parser.add_option('--lambda-min',        dest='lambda_min',      default=None,   type='float',   help='lower tidal prior bound')
    parser.add_option('--lambda-max',        dest='lambda_max',      default=None,   type='float',   help='upper tidal prior bound')
    parser.add_option('--use-mtot',          dest='use_mtot',     default=False,  action="store_true",    help='Perform sampling in mtot instead of mchirp, default False')

    # Extra parameters
    parser.add_option('--use-energy-angmom', dest='ej_flag',     default=False,  action="store_true",    help='include energy and angular momentum parameters')
    parser.add_option('--use-eccentricity',  dest='ecc_flag',    default=False,  action="store_true",    help='include energy and angular momentum parameters')
    parser.add_option('--e-min',             dest='e_min',   type='float',   default=None,   help='lower energy prior bound')
    parser.add_option('--e-max',             dest='e_max',   type='float',   default=None,   help='upper energy prior bound')
    parser.add_option('--j-min',             dest='j_min',   type='float',   default=None,   help='lower angular momentum prior bound')
    parser.add_option('--j-max',             dest='j_max',   type='float',   default=None,   help='upper angular momentum prior bound')
    parser.add_option('--ecc-min',           dest='ecc_min', type='float',   default=None,   help='lower eccentricity prior bound')
    parser.add_option('--ecc-max',           dest='ecc_max', type='float',   default=None,   help='upper eccentricity prior bound')

    # Optional, marginalize over phi_ref and/or time_shift
    parser.add_option('--marg-phi-ref',      dest='marg_phi_ref',        default=False,  action="store_true",   help='phi-ref marginalization flag')
    parser.add_option('--marg-time-shift',   dest='marg_time_shift',     default=False,  action="store_true",   help='time-shift marginalization flag')

    # Optional, number of PSD weights
    parser.add_option('--psd-weights',       dest='nweights',         default=0,     type='int',  help='number of PSD weight parameters per IFO, default 0')

    # ROQ options
    parser.add_option('--use-roq',           dest='roq',         default=False,  action="store_true",    help='ROQ flag')
    parser.add_option('--roq-epsilon',       dest='roq_epsilon', default=1e-6,   type='float',           help='accuracy for ROQ approximation, default 1e-6')
    parser.add_option('--roq-training',      dest='roq_points',  default=5000,   type='int',             help='number of training points for ROQ, default 5000')

    # GWBinning options
    parser.add_option('--use-binning',       dest='binning',     default=False,  action="store_true",    help='frequency binning flag')
    parser.add_option('--fiducial',          dest='fiducial',    default=None,   type='string',  help='path to parameters file for gwbinning')

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

# def get_roq_training_grid(prior, n_train):
#
#     _intrinsic_names = ['mchirp', 'mtot', 'q',                  # masses
#                         's1z', 's2z',                           # aligned spins
#                         's1', 'tilt1', 'phi_1l',                # precessing spin 1
#                         's2', 'tilt2', 'phi_2l',                # precessing spin 2
#                         'lambda1', 'lambda2',                   # tides
#                         'eccentricity', 'angmom', 'energy'      # non-circular
#                         ]
#
#     # default values for extrinsic parameters
#     _default = {'phi_ref':0., 'time_shift':0., 'cosi': 1.}
#     if 'distance' in prior.names:
#         _default['distance'] = np.min(prior.bounds[prior.names.index('distance')])
#
#     _n = []
#     _b = []
#
#     _o = {}
#
#     for i,ni in enumerate(prior.names):
#
#         if ni in _intrinsic_names:
#             _n.append(ni)
#             _b.append(prior.bounds[i])
#         else:
#             if ni in _default.keys():
#                 _o[ni] = _default[ni]
#             else:
#                 _o[ni] = np.mean(prior.bounds[i])
#
#     # check eos sampling
#     eos = sum([1 if 'eos' in ni for ni in prior.names else 0])
#     if eos:
#         _n.append('lambda1')
#         _b.append([0,5000])
#         _n.append('lambda2')
#         _b.append([0,5000])
#
#     n_per_dim = int(float(n_train)**(1./len(_n)))
#
#     return
#
# def initialize_roq(freqs, srate , seglen, noise, prior, approx, n_training_points, epsilon):
#     # set parameter
#     params  = get_roq_training_grid(prior, n_training_points)
#
#     # initialize waveform generator and compute waveforms
#     from ...obs.gw.waveform import Waveform
#     wave    = Waveform(freqs, srate , seglen, approx)
#     waves   = []
#
#     # compute PSD
#     psd     = noise.interp_psd_pad(freqs)
#
#     # compute ROQ
#     from .utils.roq import ROQ
#     roq     = ROQ(freqs, 1./psd, params, waves, epsilon)
#     x, i, w = roq.produceROQ()
#
#     return i, w

def get_likelihood_and_prior(opts):

    # get likelihood objects
    likes = []
    priors = []
    use_gw = False

    for ti in opts.tags:

        if ti == 'gw':

            # check
            if opts.binning and opts.roq:
                logger.error("Unable to set simultaneusly frequency-binning and ROQ approximation. Please choose one of the two options.")
                raise AttributeError("Unable to set simultaneusly frequency-binning and ROQ approximation. Please choose one of the two options.")

            # read arguments for likelihood and compute prior
            l_kwas, pr = initialize_gwlikelihood_kwargs(opts)

            if opts.binning:

                from .utils.binning import GWBinningLikelihood as GWLikelihood

            elif opts.roq:

                logger.error("ROQ approximation not yet implemented.")
                raise AttributeError("ROQ approximation not yet implemented.")

                # from .utils.roq import ROQGWLikelihood as GWLikelihood
                #
                # # compute ROQ weights
                # l_kwas['roq_weights'] = {}
                #
                # for ifo in opts.ifos:
                #
                #     logger.info("Computing ROQ weights for {} ...".format(ifo))
                #     inds, wghts = initialize_roq(l_kwas['freqs'][l_kwas['datas'][ifo].mask],
                #                                  l_kwas['srate'], l_kwas['seglen'],
                #                                  l_kwas['noises'][ifo], pr, l_kwas['approx'],
                #                                  opts.roq_epsilon, opts.roq_points)
                #     l_kwas['roq_weights'][ifo] = [inds,wghts]

            else:

                from .utils.model import GWLikelihood

            logger.info("Initializing GW likelihood ...")

            likes.append(GWLikelihood(**l_kwas))
            priors.append(pr)

            # set use_gw flag for proposal
            use_gw = True

        elif ti == 'kn':

            # select KN likelihood
            from .utils.model import KNLikelihood

            # read arguments for likelihood
            l_kwas, pr = initialize_knlikelihood_kwargs(opts)
            logger.info("Initializing KN likelihood ...")
            likes.append(KNLikelihood(**l_kwas))
            priors.append(pr)

        else:

            logger.error("Unknown tag {} for likelihood initialization. Please use gw, kn or a combination.".format(opts.tags[0]))
            raise ValueError("Unknown tag {} for likelihood initialization. Please use gw, kn or a combination.".format(opts.tags[0]))

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

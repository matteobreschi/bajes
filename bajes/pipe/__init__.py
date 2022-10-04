#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import os
import numpy as np

from scipy.linalg import expm, norm
from scipy import interpolate

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

def initialize_mpi_pool(mpi=None, comm=None, fast_mpi=False):
    from .utils.mpi import MPIPool
    pool = MPIPool(mpi=mpi, comm=comm, parallel_comms=fast_mpi)
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
    from multiprocessing import Pool
    pool = Pool(nprocs-1)
    close_pool = close_pool_mthr
    return  pool, close_pool

# pipeline/core methods

def parse_main_options():

    from .. import __version__, __doc__
    from ..inf import __known_samplers__
    import argparse

    name    = "bajes ({})".format(__version__)
    usage   = "python -m bajes [options]"
    parser  = argparse.ArgumentParser(prog=name, usage=usage, description=__doc__)

    # Model
    parser.add_argument('-p', '--prior',    dest='prior',           default=None,       type=str,                                   help='path to prior file (configuration file)')
    parser.add_argument('-l', '--like',     dest='like',            default=None,       type=str,                                   help='path to likelihood function (python file)')
    parser.add_argument('-I', '--inf',      dest='inf',             default=None,       type=str,                                   help='path to pickle inference')

    # I/O options
    parser.add_argument('-o', '--outdir',   dest='outdir',          default=None,       type=str,                                   help='directory for output')
    parser.add_argument('-D', '--debug',    dest='debug',           default=False,                      action="store_true",        help='use debugging mode for logger')
    parser.add_argument('-v', '--verbose',  dest='silence',         default=True,                       action="store_false",       help='activate stream handler, use this if you are running on terminal')
    parser.add_argument('--tracing',        dest='trace_memory',    default=False,                      action="store_true",        help='keep track of memory usage')
    parser.add_argument('--n-tracing',      dest='n_trace_memory',  default=25,         type=int,                                   help='number of iteration after each memory trace. Default: 25')

    # Generic run options
    parser.add_argument('-e', '--engine',   dest='engine',          default='dynesty',  type=str,                                   help='sampler engine name, available options: {}. Default: dynesty'.format(__known_samplers__))
    parser.add_argument('--priorgrid',      dest='priorgrid',       default=1000,       type=int,                                   help='number of nodes for prior interpolators (if needed)')
    parser.add_argument('--checkpoint',     dest='ncheck',          default=0,          type=int,                                   help='number of periodic checkpoints')
    parser.add_argument('--seed',           dest='seed',            default=None,       type=int,                                   help='seed for the pseudo-random chain')
    parser.add_argument('--use-slice',      dest='use_slice',       default=False,                      action="store_true",        help='use slice proposal (emcee, ptmcmc, cpnest)')
    parser.add_argument('--use-gw',         dest='use_gw',          default=False,                      action="store_true",        help='use slice proposal (emcee, ptmcmc, cpnest)')

    # Nested sampling options
    parser.add_argument('--nlive',          dest='nlive',           default=1024,       type=int,                                   help='[nest] number of live points. Default: 1024')
    parser.add_argument('--tol',            dest='tolerance',       default=0.1,        type=float,                                 help='[nest] evidence tolerance. Default: 0.1')
    parser.add_argument('--maxmcmc',        dest='maxmcmc',         default=4096,       type=int,                                   help='[nest] maximum number of mcmc iterations. Default: 4096')
    parser.add_argument('--minmcmc',        dest='minmcmc',         default=32,         type=int,                                   help='[nest] minimum number of mcmc iterations. Default: 32')
    parser.add_argument('--poolsize',       dest='poolsize',        default=2048,       type=int,                                   help='[nest] number of sample in the pool (cpnest). Default: 2048')
    parser.add_argument('--nact',           dest='nact',            default=5,          type=int,                                   help='[nest] sub-chain safe factor (dynesty). Default: 5')
    parser.add_argument('--nbatch',         dest='nbatch',          default=512,        type=int,                                   help='[nest] number of live points for batch (dynesty-dyn). Default: 512')
    parser.add_argument('--dkl',            dest='dkl',             default=0.5,        type=float,                                 help='[nest] target KL divergence (ultranest). Default: 0.5')
    parser.add_argument('--z-frac',         dest='z_frac',          default=None,       type=float,                                 help='[nest] remaining Z fraction (ultranest). Default: None')

    # MCMC options
    parser.add_argument('--nout',           dest='nout',            default=10000,      type=int,                                   help='[mcmc] number of posterior samples')
    parser.add_argument('--nwalk',          dest='nwalk',           default=256,        type=int,                                   help='[mcmc] number of parallel walkers')
    parser.add_argument('--nburn',          dest='nburn',           default=5000,       type=int,                                   help='[mcmc] numebr of burn-in iterations')
    parser.add_argument('--ntemp',          dest='ntemps',          default=8,          type=int,                                   help='[mcmc] number of tempered ensambles (ptmcmc)')
    parser.add_argument('--tmax',           dest='tmax',            default=None,       type=float,                                 help='[mcmc] maximum temperature scale (ptmcmc). Default: Inf')

    # Parallelization options
    parser.add_argument('-n', '--nprocs',   dest='nprocs',          default=0.,         type=int,                                   help='number of processes in the pool')
    parser.add_argument('--mpi',            dest='mpi',             default=False,                      action="store_true",        help='use MPI parallelization')
    parser.add_argument('--mpi-per-node',   dest='mpi_per_node',    default=0.,         type=int,                                   help='number of MPI processes per node')
    parser.add_argument('--fast-mpi',       dest='fast_mpi',        default=False,                      action="store_true",        help='enable fast MPI communication (under development)')

    return parser.parse_args()

def parse_setup_options():

    from .. import __version__, __doc__
    from ..inf import __known_samplers__
    import argparse

    usage   = "bajes_setup [options]"
    parser  = argparse.ArgumentParser(prog='bajes_setup',
                                      usage=usage,
                                      description="Initialize prior and likelihood in inf.pkl for parameter estimation.")

    # Generic run options
    parser.add_argument('--tag',               dest='tags',           default=[],        type=str,  action="append",      help='Tag for data messenger. Available options: [`gw`, `kn`].')
    parser.add_argument('--t-gps',             dest='t_gps',                             type=float,                         help='GPS time: for GW, center value of time axis (if a local version of the data is provided, has to coincide with this value for the input time axis); for KN, initial value of time axis')
    parser.add_argument('--priorgrid',         dest='priorgrid',      default=1000,      type=int,                           help='number of nodes for prior interpolators (if needed)')

    # I/O options
    parser.add_argument('-o', '--outdir',      dest='outdir',         default=None,      type=str,                        help='directory for output')
    parser.add_argument('--debug',             dest='debug',          default=False,                     action="store_true",  help='use debugging mode for logger')
    parser.add_argument('--verbose',           dest='silence',        default=True,                      action="store_false", help='activate stream handler, use this if you are running on terminal')

    # Fixed parameter options
    parser.add_argument('--fix-name',          dest='fixed_names',    default=[],        type=str,  action="append",      help='names of fixed params')
    parser.add_argument('--fix-value',         dest='fixed_values',   default=[],        type=float,   action="append",      help='values of fixed params')

    # Distance information
    parser.add_argument('--dist-flag',         dest='dist_flag',      default='vol',     type=str,                        help='distance prior flag (options: vol, log, com, src)')
    parser.add_argument('--dist-min',          dest='dist_min',       default=[],        type=float,   action="append",      help='lower distance prior bound')
    parser.add_argument('--dist-max',          dest='dist_max',       default=[],        type=float,   action="append",      help='upper distance prior bound')

    # Time shift (from GPS time) information
    parser.add_argument('--tshift-max',        dest='time_shift_max', default=[],        type=float,   action="append",      help='upper time shift prior bound')
    parser.add_argument('--tshift-min',        dest='time_shift_min', default=[],        type=float,   action="append",      help='lower time shift prior bound')

    #
    # GW OPTIONS
    #

    # Data and PSDs information
    parser.add_argument('--ifo',               dest='ifos',           default=[],        type=str,  action="append",      help='Detector to be considered in the analysis. Has to be passed once per detector, and sets the order for similar commands to pass strains and psds. Available options: [`H1`, `L1`, `V1`, `K1`, `G1`].')
    parser.add_argument('--strain',            dest='strains',        default=[],        type=str,  action="append",      help='Path to strain data. Has to be passed once per detector and in the same order as the `--ifo` options.')
    parser.add_argument('--asd',               dest='asds',           default=[],        type=str,  action="append",      help='Path to ASD data. Has to be passed once per detector and in the same order as the `--ifo` options.')
    parser.add_argument('--alpha',             dest='alpha',          default=None,      type=float,                         help='Alpha parameter of the Tukey window. Default: 0.4/seglen.')

    # Calibration envelopes (optional)
    parser.add_argument('--spcal',             dest='spcals',         default=[],        type=str,  action="append",      help='Path to calibration envelope. Has to be passed once per detector and in the same order as the `--ifo` options.')
    parser.add_argument('--nspcal',            dest='nspcal',         default=0,         type=int,                           help='Number of spectral calibration nodes. Default: 0')

    # Time series information
    parser.add_argument('--f-min',             dest='f_min',          default=None,      type=float,                         help='Minimum frequency at which likelihood is evaluated [Hz]. Default: None')
    parser.add_argument('--f-max',             dest='f_max',          default=None,      type=float,                         help='Maximum frequency at which likelihood is evaluated [Hz]. Default: None')
    parser.add_argument('--srate',             dest='srate',          default=None,      type=float,                         help='Rsequested sampling rate [Hz]. If smaller than data sampling rate, downsampling is applied. Default: None')
    parser.add_argument('--seglen',            dest='seglen',         default=None,      type=float,                         help='Requested length of the segment to be analysed [sec]. If smaller than data total lenght, data are cropped. If longer, data are padded. Default: None.')

    # Waveform model
    parser.add_argument('--approx',            dest='approx',         default=None,      type=str,                        help='Gravitational-wave approximant. Default: None')
    parser.add_argument('--extra-option',      dest='extra_opt',      default=[],        type=str,  action="append",      help='Names of the additional parameters for the chosen approximant. Has to be passed once for each parameter. Default: []')
    parser.add_argument('--extra-option-val',  dest='extra_opt_val',  default=[],        type=str,  action="append",      help='Values of the additional parameters for the chosen approximant. Has to be passed once for each parameter and in the same order as the `--extra-option` option. Default: []')
    parser.add_argument('--lmax',              dest='lmax',           default=0,         type=int,                           help='Higher angular mode index to be considered for GW template. The zero value corresponds to MISSING. Default: 0')
    parser.add_argument('--Eprior',            dest='Eprior',         default=None,      type=str,                           help='Prior on the initial energy in hyperbolic TEOB models. Default: None. Available options: [Constrained, Unconstrained].')
    parser.add_argument('--nqc-TEOBHyp',       dest='nqc_TEOBHyp',    default=1,         type=int,                           help='Flag to activate/deactivate (1/0) NQCs in hyperbolic TEOB models. Default: 1.')

    # Prior flags
    parser.add_argument('--data-flag',         dest='data_flag',      default=None,       type=str,                       help='Data flag. Available options: [???]. Default: None')
    parser.add_argument('--spin-flag',         dest='spin_flag',      default='no-spins', type=str,                       help='Spin prior flag. Available options: [???]. Default: `no-spins`.')
    parser.add_argument('--tidal-flag',        dest='lambda_flag',    default='no-tides', type=str,                       help='Tidal prior flag. Available options: [???]. Default: `no-tides`.')

    # Prior bounds
    parser.add_argument('--mc-min',            dest='mchirp_min',      default=None,      type=float,                        help='Lower mchirp prior bound (if use-mtot, lower mtot prior bound)')
    parser.add_argument('--mc-max',            dest='mchirp_max',      default=None,      type=float,                        help='Upper mchirp prior bound (if use-mtot, upper mtot prior bound)')
    parser.add_argument('--q-max',             dest='q_max',           default=None,      type=float,                        help='Upper mass ratio prior bound')
    parser.add_argument('--q-min',             dest='q_min',           default=1.,        type=float,                        help='Lower mass ratio prior bound')
#    parser.add_argument('--mass-max',          dest='mass_max',        default=None,      type=float,                        help='upper mass component prior bound')
#    parser.add_argument('--mass-min',          dest='mass_min',        default=None,      type=float,                        help='lower mass component prior bound')
    parser.add_argument('--spin-max',          dest='spin_max',        default=None,      type=float,                        help='Upper spin prior bound')
    parser.add_argument('--spin1-max',         dest='spin1_max',       default=None,      type=float,                        help='Upper spin1 prior bound')
    parser.add_argument('--spin2-max',         dest='spin2_max',       default=None,      type=float,                        help='Upper spin2 prior bound')
    parser.add_argument('--lambda-min',        dest='lambda_min',      default=None,      type=float,                        help='Lower tidal prior bound')
    parser.add_argument('--lambda-max',        dest='lambda_max',      default=None,      type=float,                        help='Upper tidal prior bound')
    parser.add_argument('--use-mtot',          dest='use_mtot',        default=False,                     action="store_true", help='Perform sampling in mtot instead of mchirp, default False')

    # Extra parameters
    parser.add_argument('--use-energy-angmom', dest='ej_flag',         default=False,                     action="store_true",  help='Include energy and angular momentum parameters')
    parser.add_argument('--use-eccentricity',  dest='ecc_flag',        default=False,                     action="store_true",  help='Include eccentricity parameter')
    parser.add_argument('--e-min',             dest='e_min',           default=None,      type=float,                           help='Lower energy prior bound')
    parser.add_argument('--e-max',             dest='e_max',           default=None,      type=float,                           help='Upper energy prior bound')
    parser.add_argument('--j-min',             dest='j_min',           default=None,      type=float,                           help='Lower angular momentum prior bound')
    parser.add_argument('--j-max',             dest='j_max',           default=None,      type=float,                           help='Upper angular momentum prior bound')
    parser.add_argument('--ecc-min',           dest='ecc_min',         default=None,      type=float,                           help='Lower eccentricity prior bound')
    parser.add_argument('--ecc-max',           dest='ecc_max',         default=None,      type=float,                           help='Upper eccentricity prior bound')

    # Optional, marginalize over phi_ref and/or time_shift
    parser.add_argument('--marg-phi-ref',      dest='marg_phi_ref',    default=False,                     action="store_true",  help='Phi-ref marginalization flag')
    parser.add_argument('--marg-time-shift',   dest='marg_time_shift', default=False,                     action="store_true",  help='Time-shift marginalization flag')

    # Optional, number of PSD weights
    parser.add_argument('--psd-weights',       dest='nweights',        default=0,         type=int,                             help='Number of PSD weight parameters per IFO, default 0')

    # ROQ options
    parser.add_argument('--use-roq',           dest='roq',             default=False,                     action="store_true",  help='ROQ flag, activates ROQ likelihood computation.')
    parser.add_argument('--roq-path',          dest='roq_path',        default='',        type=str,                             help='ROQ data path, storing the basis data generated following the conventions of `https://github.com/bernuzzi/PyROQ/tree/master/PyROQ`.')
    parser.add_argument('--roq-tc-points',     dest='roq_tc_points',   default=10000,     type=int,                             help='Number of points used to interpolate the time axis when using the ROQ. Default: 10000')

    # GWBinning options
    parser.add_argument('--use-binning',       dest='binning',         default=False,                     action="store_true",  help='Frequency binning flag')
    parser.add_argument('--fiducial',          dest='fiducial',        default=None,      type=str,                             help='Path to parameters file for gwbinning')

    #
    # KN OPTIONS
    #

    # Data & Components information
    parser.add_argument('--comp',         dest='comps',       type=str,  action="append", default=[],    help='Name of shell component(s) for lightcurve estimation')
    parser.add_argument('--mag-folder',   dest='mag_folder',  type=str,  default=None,    help='Path to magnitudes data folder')

    # Photometric bands information
    parser.add_argument('--band',         dest='bands',       type=str,  action="append",    default=[], help='Name of photometric bands used in the data')
    parser.add_argument('--lambda',       dest='lambdas',     type=float,   action="append",    default=[], help='Wave-length of photometric bands used in the data [nm]')
    parser.add_argument('--use-dereddening',  dest='dered',   default=True,  action="store_true",    help='Apply deredding to given data filters')

    # Prior bounds
    parser.add_argument('--mej-max',      dest='mej_max',     type=float,   action="append",    default=[], help='Upper bounds for ejected mass parameters')
    parser.add_argument('--mej-min',      dest='mej_min',     type=float,   action="append",    default=[], help='Lower bounds for ejected mass parameters')
    parser.add_argument('--vel-max',      dest='vel_max',     type=float,   action="append",    default=[], help='Upper bounds for velocity parameters')
    parser.add_argument('--vel-min',      dest='vel_min',     type=float,   action="append",    default=[], help='Lower bounds for velocity parameters')
    parser.add_argument('--opac-max',     dest='opac_max',    type=float,   action="append",    default=[], help='Upper bounds for opacity parameters')
    parser.add_argument('--opac-min',     dest='opac_min',    type=float,   action="append",    default=[], help='Lower bounds for opacity parameters')

    # Heating factor information
    parser.add_argument('--log-eps0',     dest='log_eps_flag',    default=False,  action="store_true",   help='Uniform prior in log-epsilon0 flag')
    parser.add_argument('--eps-max',      dest='eps_max',     type=float,   default=None,       help='Upper bounds for heating factor parameter')
    parser.add_argument('--eps-min',      dest='eps_min',     type=float,   default=None,       help='Lower bounds for heating factor parameter')

    # Extra heating rate coefficients
    parser.add_argument('--sample-heating',   dest='heat_sampling',    default=False,  action="store_true",   help='Include extra heating coefficients in sampling, default False')
    parser.add_argument('--heat-alpha',       dest='heating_alpha',     type=float,   default=1.3,          help='Alpha coefficient for heating rate (default 1.3)')
    parser.add_argument('--heat-time',        dest='heating_time',     type=float,   default=1.3,           help='Time coefficient for heating rate (default 1.3)')
    parser.add_argument('--heat-sigma',       dest='heating_sigma',     type=float,   default=0.11,         help='Sigma coefficient for heating rate (default 0.11)')

    # Flags
    parser.add_argument('--use-mag-dev',        dest='use_calib_sigma_lc',  default=False,  action="store_true",    help='Include systematic deviation parameter for each band')
    parser.add_argument('--use-nr-ejecta-fit',  dest='use_nr_ejecta_fit',   default=False,  action="store_true",    help='Use NR fits for dynamical ejecta (first comp.) and baryonic wind (second comp.). This option works only with joint GW-KN inference.')

    # Integrators properties
    parser.add_argument('--nvel',         dest='n_v',         type=int,     default=400,        help='Number of elements in velocity array, default 400')
    parser.add_argument('--vel-min-grid', dest='vgrid_min',   type=float,   default=1.e-7,      help='Lower limit for velocity integration, default 1e-7')
    parser.add_argument('--ntime',        dest='n_t',         type=int,     default=400,        help='Number of elements in time array, default 400')
    parser.add_argument('--t-start-grid', dest='init_t',      type=float,   default=1.,         help='Initial value of time axis for model evaluation, default 1s')
    parser.add_argument('--t-scale',      dest='t_scale',     type=str,  default='linear',   help='Scale of time axis: linear, log or mixed')

    return parser.parse_args()

# initialization functions for run

def read_model_from_paths(prior_path, like_path, priorgrid):

    import importlib, sys

    try:
        import configparser
    except ImportError:
        import ConfigParser as configparser

    from ..inf import Prior, Parameter, Likelihood, Sampler

    # get likelihood module
    logger.info("Importing likelihood from module ({}) ...".format(like_path))
    sys.path.append(os.path.dirname(like_path))
    like_module = importlib.import_module(os.path.basename(like_path).split('.')[0])

    # initialize likelihood
    try:
        lk = Likelihood(func = like_module.log_like)
    except Exception:
        logger.error("Unable to import log_like method from provided module ({}). Please check that a log_like method is provived.".format(like_path))
        raise RuntimeError("Unable to import log_like method from provided module ({}). Please check that a log_like method is provived.".format(like_path))

    # parse prior
    logger.info("Parsing prior options from file ({}) ...".format(prior_path))
    config = configparser.ConfigParser()
    config.optionxform = str
    config.sections()
    config.read(prior_path)

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
        args['interp_kwarg'] = {'ngrid': priorgrid}

        # check others
        for ki in list_args_keys:
            if ki not in _checked:
                args[ki] = float(args[ki])

        # append parameter to list
        params.append(Parameter(name = ni , **args))

    # initialize prior
    logger.info("Initializing prior distribution ...")
    pr = Prior(params)

    return pr, lk

def read_model_from_pickle(file):

    # get inference pickle
    logger.info("Importing model from container ({}) ...".format(file))
    file    = os.path.abspath(file)

    # get data container information
    from .utils import data_container
    dc = data_container(file)
    dc = dc.load()

    for pi in dc.prior.parameters:
        logger.info("Setting {} with {} prior in range [{:.3g},{:.3g}] ...".format(pi.name, pi._kind, pi.min, pi.max))

    return dc.prior, dc.like

def init_model(opts):

    # prior, likelihood and inference are None
    if (opts.prior is None) and (opts.like is None) and (opts.inf is None):
        # look for inf.pkl in outdir
        if os.path.exists(opts.outdir+'/inf.pkl'):
            pr, lk  = read_model_from_pickle(opts.outdir+'/inf.pkl')
        # else error
        else:
            logger.error("Unable to initialize model. None is given.")
            raise ValueError("Unable to initialize model. None is given.")

    # if inference pickle is None
    elif (opts.inf is None):

        if (opts.prior is None):
            logger.error("Unable to initialize model. Prior is not given.")
            raise ValueError("Unable to initialize model. Prior is not given.")

        if (opts.like is None):
            logger.error("Unable to initialize model. Likelihood is not given.")
            raise ValueError("Unable to initialize model. Likelihood is not given.")

        # get prior and likelihood from input files
        opts.prior  = os.path.abspath(opts.prior)
        opts.like   = os.path.abspath(opts.like)

        # define likelihood and prior
        pr, lk      = read_model_from_paths(opts.prior, opts.like, opts.priorgrid)

    else:

        # get inference pickle
        pr, lk      = read_model_from_pickle(opts.inf)

    return pr, lk

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

def initialize_roq(roq_path   ,
                   time_points,
                   freqs_full ,
                   data_f     ,
                   psd        ,
                   f_min      ,
                   f_max      ,
                   seglen     ,
                   prior      ,
                   approx     ):

    from .utils.roq import Initialise_pyROQ_Jena_for_inference

    # The ROQ linear weights need to be computed for all the possible coalescence times (tc).
    # Our strategy will be to split the tc bounds in N intevals, and compute the weights for the central value of each interval.
    # Then, from this set we will create an interpolant that will allow the ROQ weights computation for all possible values of tc.
    if not 'time_shift' in prior.const.keys():
        idx_t_shift = None
        for i,param in enumerate(prior.names):
            if(param=='time_shift'):
                idx_t_shift = i
                break
        t_low, t_high = prior.bounds[idx_t_shift][0], prior.bounds[idx_t_shift][1]
    else:
        t_low, t_high = prior.const['time_shift'],    prior.const['time_shift']

    # We need to account for the fact that tc will be different in each detector, so we pad the time bounds by the largest possible time delay across the Earth, 26ms, plus an additional bufer of 4ms to avoid interpolation errors at the boundaries.
    # The waveform is generated by default with the peak at tc=0.0 and then shifted to the center of the time axis, seglen/2, which corresponds to t_gps.
    central_tcs = np.linspace(seglen/2.+t_low-0.030, seglen/2.+t_high+0.030, time_points)

    # Notation and method follow `arXiv:1604.08253`.
    # Initialise ROQ evaluation class, checking that the requested parameters are covered by the constructed basis.
    # Compute the new frequency axex as a mask to be placed on top of the input array of frequencies.
    linear_frequencies, quadratic_frequencies, linear_weights_interp, quadratic_weights = Initialise_pyROQ_Jena_for_inference(path       = roq_path   ,
                                                                                                                              approx     = approx     ,
                                                                                                                              f_min      = f_min      ,
                                                                                                                              f_max      = f_max      ,
                                                                                                                              df         = 1./seglen  ,
                                                                                                                              freqs_full = freqs_full ,
                                                                                                                              psd        = psd        ,
                                                                                                                              data_f     = data_f     ,
                                                                                                                              tcs        = central_tcs)

    # Here we join the two axes together, and their masks, so that later the waveform can be evaluated just once,
    # but the distinct axes can be retrieved when computing the (h|h) or (d|h) terms.
    freqs_conc         = np.concatenate((quadratic_frequencies,linear_frequencies))
    # Avoid duplicates
    freqs_conc_no_dupl = np.array(list(set(freqs_conc)))
    # Restore order
    roq_freqs_join     = np.sort(freqs_conc_no_dupl)

    # Store masks that will be needed to evaluate the different likelihood components.
    roq_mask_lin       = [i for i in range(len(roq_freqs_join)) if roq_freqs_join[i] in linear_frequencies]
    roq_mask_qua       = [i for i in range(len(roq_freqs_join)) if roq_freqs_join[i] in quadratic_frequencies]

    return roq_freqs_join, roq_mask_qua, roq_mask_lin, quadratic_weights, linear_weights_interp

# main setup

def get_likelihood_and_prior(opts):

    # Get likelihood and prior objects for each of the "messengers" considered.
    likes  = []
    priors = []

    # Loop over the "messengers"
    for ti in opts.tags:

        if ti == 'gw':

            # Check inputs compatibility
            if opts.binning and opts.roq:
                logger.error("Unable to set simultaneusly frequency-binning and the ROQ approximation. Please choose one of the two options.")
                raise AttributeError("Unable to set simultaneusly frequency-binning and the ROQ approximation. Please choose one of the two options.")
            if opts.nspcal and opts.roq:
                logger.error("The ROQ approximation has not yet been extended to incorporate calibration uncertainties. Please choose one of the two options.")
                raise AttributeError("The ROQ approximation has not yet been extended to incorporate calibration uncertainties. Please choose one of the two options.")
            if(opts.roq_tc_points < 4):
                logger.error("Number of points on the time grid has to be at least four.")
                raise ValueError("Number of points on the time grid has to be at least four.")

            # read arguments for likelihood and compute prior
            l_kwas, pr = initialize_gwlikelihood_kwargs(opts)

            if opts.binning:

                from .utils.binning import GWBinningLikelihood as GWLikelihood

            else:

                from .log_like import GWLikelihood

                if opts.roq:

                    # compute ROQ weights
                    l_kwas['roq'] = {ifo: {} for ifo in opts.ifos}

                    for ifo in opts.ifos:

                        f_min_max_mask = l_kwas['datas'][ifo].mask
                        freqs_full     = l_kwas['freqs'][f_min_max_mask]
                        data_freq      = l_kwas['datas'][ifo].freq_series[f_min_max_mask]
                        psd            = l_kwas['noises'][ifo].interp_psd_pad(freqs_full)

                        logger.info("Computing ROQ weights for the {} detector.".format(ifo))
                        roq_freqs_join, roq_mask_psi, roq_mask_omega, roq_psi_weights, roq_omega_weights_interp = initialize_roq(opts.roq_path     ,
                                                                                                                                 opts.roq_tc_points,
                                                                                                                                 freqs_full        ,
                                                                                                                                 data_freq         ,
                                                                                                                                 psd               ,
                                                                                                                                 l_kwas['f-min']   ,
                                                                                                                                 l_kwas['f-max']   ,
                                                                                                                                 l_kwas['seglen']  ,
                                                                                                                                 pr                ,
                                                                                                                                 l_kwas['approx']  )

                        l_kwas['roq'][ifo]['omega_weights_interp'] = roq_omega_weights_interp
                        l_kwas['roq'][ifo]['psi_weights']          = roq_psi_weights

                    # These quantities are detector independent, since in bajes the time axis is the same for all detectors.
                    l_kwas['roq']['freqs_join'] = roq_freqs_join
                    l_kwas['roq']['mask_psi']   = roq_mask_psi
                    l_kwas['roq']['mask_omega'] = roq_mask_omega

            logger.info("Initializing GW likelihood ...")

            likes.append(GWLikelihood(**l_kwas))
            priors.append(pr)

        elif ti == 'kn':

            # select KN likelihood
            from .log_like import KNLikelihood

            # read arguments for likelihood
            l_kwas, pr = initialize_knlikelihood_kwargs(opts)
            l_kwas['priors'] = pr
            logger.info("Initializing KN likelihood ...")
            likes.append(KNLikelihood(**l_kwas))
            priors.append(pr)

        else:

            logger.error("Unknown tag {} for likelihood initialization. Please use gw, kn or a combination.".format(opts.tags[0]))
            raise ValueError("Unknown tag {} for likelihood initialization. Please use gw, kn or a combination.".format(opts.tags[0]))

    # Either reduce likelihood and prior objects to a single "messenger", or initialise joint objects.
    if len(opts.tags) == 0:
        logger.error("Unknown tag for likelihood initialization. Please use gw, kn or a combination.")
        raise ValueError("Unknown tag for likelihood initialization. Please use gw, kn or a combination.")

    elif len(opts.tags) == 1:
        l_obj = likes[0]
        p_obj = priors[0]

    else:

        from ..inf.prior import JointPrior
        from ..inf.likelihood import JointLikelihood

        l_kwas          = {}
        l_kwas['likes'] = likes

        p_obj = JointPrior(priors=priors)
        l_obj = JointLikelihood(**l_kwas)

    # Save the prior and likelihood objects in a pickle file, for future resuming and reproducibility.
    cont_kwargs = {'prior': p_obj, 'like': l_obj}
    save_container(opts.outdir+'/inf.pkl', cont_kwargs)

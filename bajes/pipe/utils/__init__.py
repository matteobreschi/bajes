#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import os
import numpy as np

from scipy.special import logsumexp

try:
    import pickle
except ImportError:
    import cPickle as pickle

# logger

import logging
logger = logging.getLogger(__name__)

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

    # identify master process
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception:
        rank = 0

    # save container
    if rank == 0:
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

### USE AS: pickle.load(f) -> CustomUnpickler(f).load()
### TODO: to improve and generalize, for a safer load
# class CustomUnpickler(pickle.Unpickler):
#
#     def find_class(self, module, name):
#         if name == 'erfinv':
#             from scipy.special import erfinv
#             return erfinv
#         elif name == 'erf':
#             from scipy.special import erf
#             return erf
#         return super().find_class(module, name)

class data_container(object):
    """
        Object for storing MCMC Inference class,
        It restores all the settings from previous iterations.
    """

    def __init__(self, filename):
        # initialize with a filename
        self.__file__ = os.path.abspath(filename)

    def store(self, name, data):
        # include data objects in this class
        self.__dict__[name] = data

    def save(self):

        # check stored objects
        if os.path.exists(self.__file__):
            _stored = self.load()

            # join old and new data if the container is not empty
            if _stored is not None:
                _old            = {ki: _stored.__dict__[ki] for ki in _stored.__dict__.keys() if ki not in self.__dict__.keys()}
                self.__dict__   = {**self.__dict__, **_old}

        # save objects into filename
        f = open(self.__file__, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        # load from existing filename
        f = open(self.__file__, 'rb')

        try:

            # get data
            n = pickle.load(f)
            f.close()
            return n

        except Exception as e:
            f.close()
            logger.warning("Exception ({}) occurred while loading {}. Returning empty container.".format(e, self.__file__))
            return None

# MPI helpers
def check_mpi_world(exit=True):

    import sys
    from .mpi import get_mpi_world

    try:
        mpi = get_mpi_world()
    except ImportError:
        # if ImportError, most likely mpi4py is not installed and we assume MPI is not used
        mpi = None

    if mpi is not None:
        if ((mpi.COMM_WORLD.rank > 0) and exit):
            sys.exit()

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

def log_prior_massratio(x, q_max, q_min=1.):
    from scipy.special import hyp2f1
    n  = 5.*(hyp2f1(-0.4, -0.2, 0.8, -q_min)/(q_min**0.2)-hyp2f1(-0.4, -0.2, 0.8, -q_max)/(q_max**0.2))
    return 0.4*np.log((1.+x)/(x**3.))-np.log(np.abs(n))

def log_prior_massratio_usemtot(x, q_max, q_min=1.):
    n = 1./(1.+q_min) - 1./(1.+q_max)
    return -2.*np.log(1.+x)-np.log(np.abs(n))

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
    from ...inf.prior import Parameter, Variable, Constant
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

# SNRs
def extract_snr(ifos, detectors, hphc, pars, domain, marg_phi=False, marg_time=False, ngrid=500, roq=None):

    # compute SNR and (dt, dphi) sample
    if (marg_time and marg_phi) :
        phiref, tshift, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det = extract_snr_sample_phi_time_marg(ifos, detectors, hphc, pars, domain, ngrid=ngrid)
    elif marg_time :
        tshift, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det         = extract_snr_sample_time_marg(ifos, detectors, hphc, pars, domain)
        phiref = 0.
    elif marg_phi :
        phiref, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det         = extract_snr_sample_phi_marg(ifos, detectors, hphc, pars, domain, ngrid=ngrid)
        tshift = 0.
    else :
        snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det                 = extract_snr_sample(ifos, detectors, hphc, pars, domain, roq)
        phiref = 0.
        tshift = 0.

    return phiref, tshift, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det

def extract_snr_sample(ifos, detectors, hphc, pars, domain, roq):

    dh_list     = []
    hh_list     = []

    for ifo in ifos:
        this_dh, this_hh, this_dd = detectors[ifo].compute_inner_products(hphc, pars, domain, roq)
        dh_list.append(this_dh)
        hh_list.append(this_hh)

    snr_mf2         = 0
    snr_mf_per_det  = {ifo : 0 for ifo in ifos}
    snr_opt2        = 0
    snr_opt_per_det = {ifo : 0 for ifo in ifos}

    for ifo, dh, hh in zip(ifos, dh_list, hh_list):

        # In the ROQ case, the sum was already taken when computing the scalar product with the weights.
        if roq is not None : _dh = np.real(dh)
        else               : _dh = np.real(np.sum(dh))

        snr_mf                   = _dh/np.sqrt(np.real(hh))
        snr_mf2                  = snr_mf2 + snr_mf**2
        snr_mf_per_det[ifo]      = snr_mf

        snr_opt                  = np.sqrt(np.real(hh))
        snr_opt2                 = snr_opt2 + snr_opt**2
        snr_opt_per_det[ifo]     = snr_opt

    snr_mf  = np.sqrt(snr_mf2)
    snr_opt = np.sqrt(snr_opt2)

    return snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det

def extract_snr_sample_phi_time_marg(ifos, detectors, hphc, pars, domain, ngrid=500):
    """
        Extract (time-shift,phi-ref) sample assuming (time-shift,phi-ref) are the only marginalized parameters
    """

    phi_ax      = np.linspace(0,2.*np.pi,ngrid)
    d_inner_h   = 0. + 0.j
    h_inner_h   = 0.

    dh_list     = []
    hh_list     = []

    for ifo in ifos:
        this_dh, this_hh, this_dd = detectors[ifo].compute_inner_products(hphc, pars, domain)
        d_inner_h = d_inner_h + this_dh
        h_inner_h = h_inner_h + this_hh
        dh_list.append(this_dh)
        hh_list.append(this_hh)

    # compute likelihood (for this samples) as a function of time_shift
    dh_fft   = np.fft.fft( d_inner_h )
    time_ax  = np.fft.fftfreq(len(dh_fft), d=1./pars["seglen"])
    isort    = np.argsort(time_ax)
    time_ax  = time_ax[isort]
    dh_fft   = dh_fft[isort]

    # compute likelihood (for this sample) as a function of time_shift and phi_ref
    # NOTE: like_mat is a rank-2 matrix where like_mat[i][j] is the loglike for time_ax[i] and phi_ax[j]
    like_mat = -0.5*h_inner_h + np.real([ dh * np.exp(-1j*phi_ax) for dh in dh_fft ])

    # sum over phi_ax and extract time_shift
    like_arr    = logsumexp(like_mat, axis=1)
    prob        = np.exp(like_arr - logsumexp(like_arr))
    cdf         = np.cumsum(prob)
    cdf         /= np.max(cdf)
    time_shift  = np.interp(np.random.uniform(0,1), cdf, time_ax)

    # identify time_shift on time_ax and compute likelihood as a function of phi_ref
    if time_shift in time_ax:
        ix = np.argmin(np.abs(time_shift-time_ax))
        like_arr = np.array(like_mat[ix])
    else:
        # NOTE: there are only two values that satify
        # |time_shift-time_ax| < dt
        # if time_shift is not in time_ax
        ix  = np.where(np.abs(time_shift-time_ax)<1./pars['srate'])
        iup = np.min(ix)
        ilo = np.max(ix)
        tup = time_ax[iup]
        tlo = time_ax[ilo]
        Lup = like_mat[iup]
        Llo = like_mat[ilo]
        like_arr = np.array( [ np.interp(time_shift, [tlo,tup], [Llo[i],Lup[i]]) for i in range(len(Lup)) ] )

    # extract phi_ref
    prob    = np.exp(like_arr - logsumexp(like_arr))
    cdf     = np.cumsum(prob)
    cdf     /= np.max(cdf)
    phi_ref = np.interp(np.random.uniform(0,1), cdf, phi_ax)

    # compute SNR

    snr_mf2         = 0
    snr_mf_per_det  = {ifo : 0 for ifo in ifos}
    snr_opt2        = 0
    snr_opt_per_det = {ifo : 0 for ifo in ifos}


    f_ax      = np.linspace(0, pars['srate']/2, int(pars['srate']*pars['seglen']//2+1))
    unit_fact = np.exp(-1j* ( phi_ref + 2*np.pi*time_shift*f_ax ) )

    for ifo, dh, hh in zip(ifos, dh_list, hh_list):

        _dh                  = np.real(np.sum(dh*unit_fact))

        snr_mf               = _dh/np.sqrt(np.real(hh))
        snr_mf2              = snr_mf2 + snr_mf**2
        snr_mf_per_det[ifo]  = snr_mf

        snr_opt              = np.sqrt(np.real(hh))
        snr_opt2             = snr_opt2 + snr_opt**2
        snr_opt_per_det[ifo] = snr_opt

    snr_mf  = np.sqrt(snr_mf2)
    snr_opt = np.sqrt(snr_opt2)

    return phi_ref, time_shift, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det

def extract_snr_sample_time_marg(ifos, detectors, hphc, pars, domain):
    """
        Extract time-shift sample assuming time-shift is the only marginalized parameter
    """

    d_inner_h   = 0. + 0.j
    h_inner_h   = 0.

    dh_list     = []
    hh_list     = []

    for ifo in ifos:
        this_dh, this_hh, this_dd = detectors[ifo].compute_inner_products(hphc, pars, domain)
        d_inner_h = d_inner_h + this_dh
        h_inner_h = h_inner_h + this_hh
        dh_list.append(this_dh)
        hh_list.append(this_hh)

    # See https://dcc.ligo.org/LIGO-T1400460
    # compute likelihood (for this sample) as a function of time_shift
    like_arr = -0.5*h_inner_h + np.real(np.fft.fft( d_inner_h ))
    time_ax  = np.fft.fftfreq(len(like_arr), d=1./pars["seglen"])
    isort    = np.argsort(time_ax)
    time_ax  = time_ax[isort]
    like_arr = like_arr[isort]

    # extract time_shift random sample
    prob    = np.exp(like_arr - logsumexp(like_arr))
    cdf     = np.cumsum(prob)
    cdf     /= np.max(cdf)
    time_shift = np.interp(np.random.uniform(0,1), cdf, time_ax)

    # compute SNR
    snr_mf2         = 0
    snr_mf_per_det  = {ifo : 0 for ifo in ifos}
    snr_opt2        = 0
    snr_opt_per_det = {ifo : 0 for ifo in ifos}

    f_ax      = np.linspace(0, pars['srate']/2, int(pars['srate']*pars['seglen']//2+1))
    unit_fact = np.exp( -2j*np.pi*time_shift*f_ax )
    for ifo, dh, hh in zip(ifos, dh_list, hh_list):

        _dh                  = np.real(np.sum(dh*unit_fact))

        snr_mf               = _dh/np.sqrt(np.real(hh))
        snr_mf2              = snr_mf2 + snr_mf**2
        snr_mf_per_det[ifo]  = snr_mf

        snr_opt              = np.sqrt(np.real(hh))
        snr_opt2             = snr_opt2 + snr_opt**2
        snr_opt_per_det[ifo] = snr_opt

    snr_mf  = np.sqrt(snr_mf2)
    snr_opt = np.sqrt(snr_opt2)

    return time_shift, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det

def extract_snr_sample_phi_marg(ifos, detectors, hphc, pars, domain, ngrid=500):
    """
        Extract phi-ref sample assuming phi-ref is the only marginalized parameter
    """

    phi_ax      = np.linspace(0,2.*np.pi,ngrid)
    d_inner_h   = 0. + 0.j
    h_inner_h   = 0.

    dh_list     = []
    hh_list     = []

    for ifo in ifos:
        this_dh, this_hh, this_dd = detectors[ifo].compute_inner_products(hphc, pars, domain)
        d_inner_h = d_inner_h + this_dh.sum()
        h_inner_h = h_inner_h + this_hh
        dh_list.append(this_dh)
        hh_list.append(this_hh)

    # compute likelihood (for this sample) as a function of phi_ref
    # See https://dcc.ligo.org/LIGO-T1300326
    like_arr = -0.5*h_inner_h + np.real( d_inner_h * np.exp(-1j*phi_ax) )

    # extract phi_ref random sample
    prob    = np.exp(like_arr - logsumexp(like_arr))
    cdf     = np.cumsum(prob)
    cdf     /= np.max(cdf)
    phi_ref = np.interp(np.random.uniform(0,1), cdf, phi_ax)

    # compute SNR
    snr_mf2         = 0
    snr_mf_per_det  = {ifo : 0 for ifo in ifos}
    snr_opt2        = 0
    snr_opt_per_det = {ifo : 0 for ifo in ifos}

    unit_fact = np.exp(-1j*phi_ref)
    for ifo, dh, hh in zip(ifos, dh_list, hh_list):

        _dh                  = np.real(np.sum(dh*unit_fact))

        snr_mf               = _dh/np.sqrt(np.real(hh))
        snr_mf2              = snr_mf2 + snr_mf**2
        snr_mf_per_det[ifo]  = snr_mf

        snr_opt              = np.sqrt(np.real(hh))
        snr_opt2             = snr_opt2 + snr_opt**2
        snr_opt_per_det[ifo] = snr_opt

    snr_mf  = np.sqrt(snr_mf2)
    snr_opt = np.sqrt(snr_opt2)

    return phi_ref, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det

# external modules helpers
def _get_astropy_version():
    from astropy import __version__ as astro_version
    return astro_version.split('.')

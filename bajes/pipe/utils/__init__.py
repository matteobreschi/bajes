#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import os
import numpy as np

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
            logger.warning("Exception ({}) occurred while loading {}.".format(e, self.__file__))
            return None

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

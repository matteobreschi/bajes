#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
import numpy as np

# logger
import logging
logger = logging.getLogger(__name__)

from .utils import *

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

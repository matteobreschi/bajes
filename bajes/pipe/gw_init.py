#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
import numpy as np

# logger
import logging
logger = logging.getLogger(__name__)

from .utils import *

def initialize_gwlikelihood_kwargs(opts):

    from ..obs.gw.noise    import Noise
    from ..obs.gw.strain   import Series
    from ..obs.gw.detector import Detector
    from ..obs.gw.utils    import read_data, read_asd, read_spcal

    # initial checks
    if len(opts.ifos) != len(opts.strains):
        logger.error("Number of IFOs {} does not match the number of data {}. Please give in input the same number of arguments in the respective order.".format(len(opts.ifos), len(opts.strains)))
        raise ValueError("Number of IFOs {} does not match the number of data {}. Please give in input the same number of arguments in the respective order.".format(len(opts.ifos), len(opts.strains)))
    elif len(opts.ifos) != len(opts.asds):
        logger.error("Number of IFOs does not match the number of ASDs. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of IFOs does not match the number of ASDs. Please give in input the same number of arguments in the respective order.")

    if (opts.f_min is None) or (opts.f_max is None) or (opts.srate is None) or (opts.seglen is None):
        logger.error("f-min, f-max, srate and seglen cannot be None. Please provide a value for all these parameters.")
        raise ValueError("f-min, f-max, srate and seglen cannot be None. Please provide a value for all these parameters.")

    if opts.f_max > opts.srate/2.:
        logger.error("Requested f_max greater than f_Nyquist=sampling_rate/2, which will induce information loss, see https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem. Please use f_max <= f_Nyquist.")
        raise ValueError("Requested f_max greater than f_Nyquist=sampling_rate/2, which will induce information loss, see https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem. Please use f_max <= f_Nyquist.")
    if opts.time_shift_min == None:
        opts.time_shift_min = -opts.time_shift_max

    # initialise dictionaries for detectors, noises, data, etc
    strains = {}
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
        data        = read_data(opts.data_flag, opts.strains[i], opts.srate)
        f_asd , asd = read_asd(opts.asds[i], ifo)

        # check ASD domain
        f_asd_min, f_asd_max = np.min(f_asd), np.max(f_asd)
        if (opts.f_min < f_asd_min) or (opts.f_max > f_asd_max):
            logger.error("The provided ASD for the {} IFO does not have support over the full requested [f-min, f-max] = [{}, {}] range. While, the ASD has [{}, {}]".format(ifo, opts.f_min, opts.f_max, f_asd_min, f_asd_max))
            raise ValueError("The provided ASD for the {} IFO does not have support over the full requested [f-min, f-max] = [{}, {}] range. While, the ASD has [{}, {}]".format(ifo, opts.f_min, opts.f_max, f_asd_min, f_asd_max))

        if opts.binning:
            # if frequency binning is on, the frequency series does not need to be cut
            strains[ifo]      = Series('time' ,
                                       data ,
                                       srate=opts.srate,
                                       seglen=opts.seglen,
                                       f_min=opts.f_min,
                                       f_max=opts.f_max,
                                       t_gps=opts.t_gps,
                                       only=False,
                                       alpha_taper=opts.alpha)

        else:
            strains[ifo]      = Series('time' ,
                                       data ,
                                       srate=opts.srate,
                                       seglen=opts.seglen,
                                       f_min=opts.f_min,
                                       f_max=opts.f_max,
                                       t_gps=opts.t_gps,
                                       only=False,
                                       alpha_taper=opts.alpha)

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

    # check spins
    if opts.spin_max is None and opts.spin_flag != 'no-spins':
        # try to set individual spin priors
        if opts.spin1_max is None or opts.spin2_max is None:
            logger.error("Unable to set individual spin prior. Input value is missing.")
            raise RuntimeError("Unable to set individual spin prior. Input value is missing.")
        opts.spin_max = [opts.spin1_max,opts.spin2_max]

    # define priors
    priors, l_kwargs['spcal_freqs'], l_kwargs['len_weights'] = initialize_gwprior(opts.ifos,
                                                                                  [opts.mchirp_min,opts.mchirp_max],
                                                                                  [opts.q_min, opts.q_max],
                                                                                  opts.f_min, opts.f_max,
                                                                                  opts.t_gps,
                                                                                  opts.seglen,
                                                                                  opts.srate,
                                                                                  opts.approx,
                                                                                  freqs,
                                                                                  spin_flag=opts.spin_flag,
                                                                                  spin_max=opts.spin_max,
                                                                                  lambda_flag=opts.lambda_flag,
                                                                                  lambda_max=opts.lambda_max,
                                                                                  lambda_min=opts.lambda_min,
                                                                                  dist_flag=opts.dist_flag,
                                                                                  dist_max=opts.dist_max,
                                                                                  dist_min=opts.dist_min,
                                                                                  time_shift_bounds=[opts.time_shift_min, opts.time_shift_max],
                                                                                  fixed_names=opts.fixed_names,
                                                                                  fixed_values=opts.fixed_values,
                                                                                  extra_opt=opts.extra_opt,
                                                                                  extra_opt_val=opts.extra_opt_val,
                                                                                  spcals = spcals,
                                                                                  nspcal = opts.nspcal,
                                                                                  nweights = opts.nweights,
                                                                                  ej_flag = opts.ej_flag,
                                                                                  ecc_flag = opts.ecc_flag,
                                                                                  energ_bounds=e_bounds,
                                                                                  angmom_bounds=j_bounds,
                                                                                  ecc_bounds=ecc_bounds,
                                                                                  marg_phi_ref = opts.marg_phi_ref,
                                                                                  marg_time_shift = opts.marg_time_shift,
                                                                                  tukey_alpha = opts.alpha,
                                                                                  lmax = opts.lmax,
                                                                                  Eprior = opts.Eprior,
                                                                                  nqc_TEOBHyp = opts.nqc_TEOBHyp,
                                                                                  prior_grid=opts.priorgrid,
                                                                                  kind='linear',
                                                                                  use_mtot=opts.use_mtot)


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

def initialize_gwprior(ifos,
                       mchirp_bounds,
                       q_bounds,
                       f_min, f_max,
                       t_gps,
                       seglen,
                       srate,
                       approx,
                       freqs,
                       spin_flag='no-spins', spin_max=None,
                       lambda_flag='no-tides', lambda_max=None, lambda_min=None,
                       dist_flag='vol', dist_max=None, dist_min=None,
                       time_shift_bounds=None,
                       fixed_names=[], fixed_values=[],
                       extra_opt =[], extra_opt_val=[],
                       spcals=None, nspcal=0, nweights=0,
                       ej_flag = False, ecc_flag = False,
                       energ_bounds=None, angmom_bounds=None, ecc_bounds=None,
                       marg_phi_ref=False, marg_time_shift=False,
                       tukey_alpha=None,
                       lmax=2,
                       Eprior = None,
                       nqc_TEOBHyp = 1,
                       prior_grid=2000,
                       kind='linear',
                       use_mtot=False):

    from ..inf.prior import Prior, Parameter, Variable, Constant

    names   = []
    bounds  = []
    funcs   = []
    kwargs  = {}

    interp_kwarg = {'ngrid': prior_grid, 'kind': kind}

    # wrap everything into a dictionary
    dict = {}

    if use_mtot:
        # setting masses (mtot,q)
        dict['mtot']    = Parameter(name='mtot',
                                    min=mchirp_bounds[0],
                                    max=mchirp_bounds[1],
                                    prior='linear')

        dict['q']       = Parameter(name='q',
                                    min=q_bounds[0],
                                    max=q_bounds[1],
                                    func=log_prior_massratio_usemtot,
                                    func_kwarg={'q_max': q_bounds[1],
                                                'q_min': q_bounds[0]},
                                    interp_kwarg=interp_kwarg)

    else:
        # setting masses (mchirp,q)
        dict['mchirp']  = Parameter(name='mchirp',
                                    min=mchirp_bounds[0],
                                    max=mchirp_bounds[1],
                                    prior='linear')

        dict['q']       = Parameter(name='q',
                                    min=q_bounds[0],
                                    max=q_bounds[1],
                                    func=log_prior_massratio,
                                    func_kwarg={'q_max': q_bounds[1],
                                                'q_min': q_bounds[0]},
                                    interp_kwarg=interp_kwarg)

    # setting spins
    if spin_max is not None:

        if isinstance(spin_max,list):
            if spin_max[0] > 1.:
                logger.warning("Input spin1-max is greater than 1, this is not a physical value. The input value will be ignored and spin1-max will be 1.")
                spin_max[0] = 1.
            if spin_max[1] > 1.:
                logger.warning("Input spin2-max is greater than 1, this is not a physical value. The input value will be ignored and spin2-max will be 1.")
                spin_max[1] = 1.

        else:
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

        if spin_max is None:
            logger.error("Spinning model requested without input maximum spin specification. Please include argument spin_max in Prior")
            raise ValueError("Spinning model requested without input maximum spin specification. Please include argument spin_max in Prior")

        elif spin_flag == 'align-volumetric':

            if isinstance(spin_max,list):

                dict['s1z'] = Parameter(name='s1z',
                                        min=-spin_max[0],
                                        max=spin_max[0],
                                        func=log_prior_spin_align_volumetric,
                                        func_kwarg={'spin_max':spin_max[0]},
                                        interp_kwarg=interp_kwarg)

                dict['s2z'] = Parameter(name='s2z',
                                        min=-spin_max[1],
                                        max=spin_max[1],
                                        func=log_prior_spin_align_volumetric,
                                        func_kwarg={'spin_max':spin_max[1]},
                                        interp_kwarg=interp_kwarg)

            else:

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

            if isinstance(spin_max,list):

                dict['s1z'] = Parameter(name='s1z',
                                        min=-spin_max[0],
                                        max=spin_max[0],
                                        func=log_prior_spin_align_isotropic,
                                        func_kwarg={'spin_max':spin_max[0]},
                                        interp_kwarg=interp_kwarg)

                dict['s2z'] = Parameter(name='s2z',
                                        min=-spin_max[1],
                                        max=spin_max[1],
                                        func=log_prior_spin_align_isotropic,
                                        func_kwarg={'spin_max':spin_max[1]},
                                        interp_kwarg=interp_kwarg)

            else:

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

            if isinstance(spin_max,list):


                dict['s1']      = Parameter(name='s1',
                                            min=0.,
                                            max=spin_max[0],
                                            prior='quadratic')
                dict['s2']      = Parameter(name='s2',
                                            min=0.,
                                            max=spin_max[1],
                                            prior='quadratic')

            else:

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

            if isinstance(spin_max,list):

                dict['s1']      = Parameter(name='s1',
                                            min=0.,
                                            max=spin_max[0],
                                            prior_func='uniform')
                dict['s2']      = Parameter(name='s2',
                                            min=0.,
                                            max=spin_max[1],
                                            prior_func='uniform')

            else:

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
            dict['eos_logp1'] = Parameter(name='eos_logp1', min=32., max=35.)
            dict['eos_gamma1'] = Parameter(name='eos_gamma1', min=1.4,  max=5.)
            dict['eos_gamma2'] = Parameter(name='eos_gamma2', min=0,    max=8.)
            dict['eos_gamma3'] = Parameter(name='eos_gamma3', min=0.5,  max=8.)

        elif lambda_flag == 'bhns-tides':
            dict['lambda1'] = Constant('lambda1', 0.)
            dict['lambda2'] = Parameter(name='lambda2',
                                        min=lambda_min,
                                        max=lambda_max,
                                        prior='uniform')

        elif lambda_flag == 'bhns-eos4p':
            dict['lambda1'] = Constant('lambda1', 0.)
            dict['eos_logp1'] = Parameter(name='eos_logp1', min=32., max=35.)
            dict['eos_gamma1'] = Parameter(name='eos_gamma1', min=1.4,  max=5.)
            dict['eos_gamma2'] = Parameter(name='eos_gamma2', min=0,    max=8.)
            dict['eos_gamma3'] = Parameter(name='eos_gamma3', min=0.5,  max=8.)

        elif lambda_flag == 'nsbh-tides':
            dict['lambda1'] = Parameter(name='lambda1',
                                        min=lambda_min,
                                        max=lambda_max,
                                        prior='uniform')
            dict['lambda2'] = Constant('lambda2', 0.)

        elif lambda_flag == 'nsbh-eos4p':
            dict['lambda2'] = Constant('lambda2', 0.)
            dict['eos_logp1'] = Parameter(name='eos_logp1', min=32., max=35.)
            dict['eos_gamma1'] = Parameter(name='eos_gamma1', min=1.4,  max=5.)
            dict['eos_gamma2'] = Parameter(name='eos_gamma2', min=0,    max=8.)
            dict['eos_gamma3'] = Parameter(name='eos_gamma3', min=0.5,  max=8.)

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
                                                              min = np.max([0, 1.-5.*sigma_w]),
                                                              max = 1.+5.*sigma_w,
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
            logger.warning("Requested bounds for energy parameter is empty. Setting standard bound [1.0001,1.1]")
            energ_bounds = [1.0001,1.1]

        dict['energy'] = Parameter(name='energy', min=energ_bounds[0], max=energ_bounds[1])

        if angmom_bounds == None:
            logger.warning("Requested bounds for angular momentum parameter is empty. Setting standard bound [3.5,4.5]")
            angmom_bounds = [3.5,4.5]

        dict['angmom'] = Parameter(name='angmom', min=angmom_bounds[0], max=angmom_bounds[1])

    # include extra parameters: eccentricity
    if ecc_flag:
        if ecc_bounds == None:
            logger.warning("Requested bounds for eccentricity parameter is empty. Setting standard bound [0,1]")
            ecc_bounds = [0., 1.]
        dict['eccentricity'] = Parameter(name='eccentricity', min=ecc_bounds[0], max=ecc_bounds[1])
    else:
        dict['eccentricity'] = Constant('eccentricity', 0.)

    # include NRPMw additional parameters
    if 'NRPMw' in approx:
        dict['NRPMw_phi_pm']    = Parameter(name='NRPMw_phi_pm',    max = 2.*np.pi, min = 0., periodic=1)   # post-merger phase [rads]
        dict['NRPMw_t_coll']    = Parameter(name='NRPMw_t_coll',    max=3000, min=1)                        # time of collapse after merger [mass-rescaled geom. units]
        dict['NRPMw_df_2']      = Parameter(name='NRPMw_df_2',      max=1e-4, min=-1e-4)                    # f_2 slope [mass-rescaled geom. units]

    # include NRPMw recalibration parameters
    if approx == 'NRPMw_recal':
        from ..obs.gw.approx.nrpmw import __recalib_names__, __ERRS__, __BNDS__
        for ni in __recalib_names__:
            dict['NRPMw_recal_'+ni] = Parameter(name='NRPMw_recal_'+ni,
                                                max = __BNDS__[ni][1], min = __BNDS__[ni][0],
                                                prior='normal', mu = 0., sigma = __ERRS__[ni])

    # include NRPMw recalibration parameters
    if approx == 'TEOBResumSPA_NRPMw_recal':
        from ..obs.gw.approx.nrpmw import __recalib_names_attach__, __ERRS__, __BNDS__
        for ni in __recalib_names_attach__:
            dict['NRPMw_recal_'+ni] = Parameter(name='NRPMw_recal_'+ni,
                                                max = __BNDS__[ni][1], min = __BNDS__[ni][0],
                                                prior='normal', mu = 0., sigma = __ERRS__[ni])

    # include NRPM recalibration and extended parameters
    if 'NRPM_ext' in approx:
        dict['NRPM_phi_pm']         = Parameter(name='NRPM_phi_pm',         max = 2.*np.pi, min = 0., periodic=1)  # post-merger phase
        dict['NRPM_alpha_inverse']  = Parameter(name='NRPM_alpha_inverse',  max = 1000,     min = 1.)              # post-merger damping time
        dict['NRPM_beta']           = Parameter(name='NRPM_beta',           max = 1e-4,     min = -1e-4)           # post-merger frequency slope

    if 'NRPM_ext_recal' in approx:
        from ..obs.gw.approx.nrpm import __recalib_names__, __ERRS__
        for ni in __recalib_names__:
            dict['NRPM_recal_'+ni] = Parameter(name='NRPM_recal_'+ni, max = 1., min = -1., prior='normal', mu = 0., sigma = __ERRS__[ni])

    # set fixed parameters
    # OBS. This step must be done when all parameters are in dict
    if len(fixed_names) != 0 :
        assert len(fixed_names) == len(fixed_values)
        for ni,vi in zip(fixed_names,fixed_values) :
            if ni not in list(dict.keys()):
                logger.warning("Requested fixed parameters ({}={}) is not in the list of all parameters. The command will be ignored.".format(ni,vi))
                continue
            else:
                dict[ni] = Constant(ni, vi)

    # set the extra options for the approximants
    if len(extra_opt) !=0:
        assert len(extra_opt) == len(extra_opt_val)
        for ni,vi in zip(extra_opt,extra_opt_val) :
            dict[ni] = Constant(ni, vi)

    # fill values for the waveform and the likelihood
    dict['f_min']  = Constant('f_min',  f_min)
    dict['f_max']  = Constant('f_max',  f_max)
    dict['t_gps']  = Constant('t_gps',  t_gps)
    dict['seglen'] = Constant('seglen', seglen)
    dict['srate']  = Constant('srate',  srate)
    dict['lmax']   = Constant('lmax',   lmax)
    dict['Eprior'] = Constant('Eprior', Eprior)
    dict['nqc-TEOBHyp'] = Constant('nqc-TEOBHyp', nqc_TEOBHyp)

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

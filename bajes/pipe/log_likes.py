from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.special import i0e

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

run_settings = { 'ifos'            : ['H1', 'L1'],
                 'approx'          : 'MLGW',
                 'nspcal'          : 0,
                 'spcal_freqs'     : None,
                 'nweights'        : 0,
                 'len_weights'     : None,
                 'marg_phi_ref'    : False,
                 'marg_time_shift' : False,
                 'tukey'           : 0.1
               }

noise_dict = { 'event'    : 'GW150914',
               'settings' : { 'f_min'  : 20,
                              'f_max'  : 1024
                            }
             }

series_dict = { 'domain'   : 'time',
                'settings' : { 'seglen' : 8,
                               'srate'  : 4096,
                               't_gps'  : 1126259462,
                               'f_min'  : 20,
                               'f_max'  : 1024
                             },
                'H1'       : { 'data_path'   : 'data/H1_STRAIN_8_4096_1126259462.txt',
                               'usecols'     : [0,1],
                               'unpack'      : True,
                               'bandpassing' : {'flow'  : 20,
                                                'fhigh' : 300}
                             },
                'L1'       : { 'data_path'   : 'data/L1_STRAIN_8_4096_1126259462.txt',
                                'usecols'     : [0,1],
                                'unpack'      : True,
                                'bandpassing' : {'flow'  : 20,
                                                 'fhigh' : 300}
                             }
              }

logger.debug("Preparing network")
net = Network(run_settings['ifos'], series_dict['settings']['t_gps'])
logZ_noise = prep_net_for_log_like(noise_dict, series_dict, prep_wave(run_settings['approx']))

def prep_gw_params(bayes_params, run_settings, noise_dict, series_dict):
    return {'mchirp'     : bayes_params['mchirp'],
            'q'          : bayes_params['q'],
            's1x'        : bayes_params['s1x'],
            's1y'        : bayes_params['s1y'],
            's1z'        : bayes_params['s1z'],
            's2x'        : bayes_params['s2x'],
            's2y'        : bayes_params['s2y'],
            's2z'        : bayes_params['s2z'],
            'lambda1'    : bayes_params['lambda1'],
            'lambda2'    : bayes_params['lambda2'],
            'distance'   : bayes_params['distance'],
            'iota'       : bayes_params['iota'],
            'ra'         : bayes_params['ra'],
            'dec'        : bayes_params['dec'],
            'psi'        : bayes_params['psi'],
            'time_shift' : bayes_params['time_shift'],
            'phi_ref'    : bayes_params['phi_ref'],
            'f_min'      : noise_dict['settings']['f_min'],
            'srate'      : series_dict['settings']['srate'],
            'seglen'     : series_dict['settings']['seglen'],
            'tukey'      : run_settings['tukey'],
            't_gps'      : series_dict['settings']['t_gps']
           }

def comp_dh_hh_dd_psd_fact(bayes_params, run_settings, noise_dict, series_dict):
    logger.debug("Preparing params")
    params = prep_gw_params(bayes_params, run_settings, noise_dict, series_dict) 

    logger.debug("Evaluatoing waveform for {}".format(params))
    hphc = net.eval_wave(params)
    # if hp, hc == [None], [None] --> the requested parameters are unphysical --> return -inf
    if not any(hphc.plus): return -np.inf

    logger.debug("Computing inner_products")
    dh_arrs, hh, dd, psd_fact = net.compute_inner_products(hphc, params, psd_weight_factor=True)

def comp_dh_with_time_shift_marg(dh_arrs):
    logger.debug("Marginalizing time_shift")
    return np.sum(np.array([ np.fft.fft(dh_arr) for dh_arr in dh_arrs ]))

def comp_dh_without_time_shift_marg(dh_arrs):
    logger.debug("Not marginalizing time_shift")
    return np.sum(np.array([ dh_arr.sum()       for dh_arr in dh_arrs ]))

def comp_R_with_phi_ref_marg(dh):
    logger.debug("Marginalizing phi_ref")
    return np.log(i0e(np.abs(dh))) + np.abs(dh)

def comp_R_without_phi_ref_marg(dh):
    logger.debug("Not marginalizing phi_ref")
    return np.real(dh)

def comp_logsumexp_of_R(R, net):
    logger.debug("Computing logsumexp of R")
    return logsumexp(R - np.log(len(net.wave.freqs)))

def comp_log_like(hh, dd, R, logZ_noise, psd_fact):
    logger.debug("Estimating likelihood")
    return -0.5 * (hh + dd) + R - logZ_noise - 0.5 * psd_fact

def gw_log_like_marg_time_shift_phi_ref(bayes_params):
    dh_arrs, hh, dd, psd_fact = get_dh_hh_dd_psd_fact(bayes_params, run_settings, noise_dict, series_dict)
    dh = comp_dh_with_time_shift_marg(dh_arrs)
    R  = comp_R_with_phi_ref_marg(dh)
    R  = comp_logsumexp_of_R(R, net)
    return comp_log_like(hh, dd, R, logZ_noise, psd_fact)

def gw_log_like_marg_time_shift(bayes_params):
    dh_arrs, hh, dd, psd_fact = get_dh_hh_dd_psd_fact(bayes_params, run_settings, noise_dict, series_dict)
    dh = comp_dh_with_time_shift_marg(dh_arrs)
    R  = comp_R_without_phi_ref_marg(dh)
    R  = comp_logsumexp_of_R(R, net)
    return comp_log_like(hh, dd, R, logZ_noise, psd_fact)

def gw_log_like_marg_phi_ref(bayes_params):
    dh_arrs, hh, dd, psd_fact = get_dh_hh_dd_psd_fact(bayes_params, run_settings, noise_dict, series_dict)
    dh = comp_dh_without_time_shift_marg(dh_arrs)
    R  = comp_R_with_phi_ref_marg(dh)
    return comp_log_like(hh, dd, R, logZ_noise, psd_fact)

def gw_log_like_no_marg(bayes_params):
    dh_arrs, hh, dd, psd_fact = get_dh_hh_dd_psd_fact(bayes_params, run_settings, noise_dict, series_dict)
    dh = comp_dh_without_time_shift_marg(dh_arrs)
    R  = comp_R_without_phi_ref_marg(dh)
    return comp_log_like(hh, dd, R, logZ_noise, psd_fact)

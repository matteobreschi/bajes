from __future__ import division, unicode_literals
import numpy as np

import logging
logger = logging.getLogger(__name__)

# # Dictionary of known approximants
# # Each key corresponds to the name of the approximant
# # Each value has to be a dictionary
# # that include the following keys:
# #   * 'path':   string to method to be imported, e.g. bajes.obs.gw.approx.taylorf2.taylorf2_35pn_wrapper
# #   * 'type':   define if the passed func is a function or a class, options: ['fnc', 'cls']
# #   * 'domain': define if the method returns a frequency- or time-domain waveform, options: ['time', 'freq']

__approx_dict__ = { ### TIME-DOMAIN
                    # funcs
                    'KilonovaHeatingRate-1':                {'path': 'bajes.obs.kn.approx.kilonova_heating_rate.kilonova_heating_rate_one_wrapper',
                                                             'type': 'fnc'},
                    'KilonovaHeatingRate-2':                {'path': 'bajes.obs.kn.approx.kilonova_heating_rate.kilonova_heating_rate_two_wrapper',
                                                             'type': 'fnc'},
                    # classes
                    'GrossmanKBP-1-isotropic':              {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_isotropic_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-1-equatorial':             {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_equatorial_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-1-polar':                  {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_polar_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-2-isotropic':              {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_two_isotropic_isotropic_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-2-equatorial':             {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_two_isotropic_equatorial_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-2-polar':                  {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_two_isotropic_polar_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-2-eq+pol':                 {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_two_equatorial_polar_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-2-NRfits-iso':             {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_two_nrfit_isotropic_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-2-NRfits-ani':             {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_two_nrfit_anisotropic_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-3-isotropic':              {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_three_isotropic_wrapper',
                                                             'type': 'cls'},
                    'GrossmanKBP-3-anisotropic':            {'path': 'bajes.obs.kn.approx.grossman_kbp.korobkin_barnes_grossman_perego_et_al_three_anisotropic_wrapper',
                                                             'type': 'cls'},
                  }

def __get_lightcurve_generator__(approx, times, lambdas, **kwargs):

    # get approximant list
    __known_approxs__ = list(__approx_dict__.keys())

    # non-LAL waaveforms
    if (approx not in __known_approxs__):
        logger.error("Unable to read approximant string. Please use a valid string: {}.".format(__known_approxs__))
        raise AttributeError("Unable to read approximant string. Please use a valid string: {}.".format(__known_approxs__))

    this_light = __approx_dict__[approx]
    light_pars = {'times'   : times,
                  'lambdas' : lambdas,
                  'v_min'   : kwargs.get('v_min',   1.e-7),
                  'n_v'     : kwargs.get('n_v',     400),
                  't_start' : kwargs.get('t_start', 1),
                  }

    # set module string and import
    from importlib import import_module
    path_to_method  = this_light['path'].split('.')
    light_module    = import_module('.'.join(path_to_method[:-1]))

    # this condition never occurs if the code is properly written
    if path_to_method[-1] not in dir(light_module):
        raise AttributeError("Unable to import {} method from {}".format(path_to_method[-1], light_module))

    # get waveform generator and domain string
    if this_light['type'] == 'fnc':
        light_func = getattr(light_module, path_to_method[-1])
    elif this_light['type'] == 'cls':
        light_obj  = getattr(light_module, path_to_method[-1])
        light_func = light_obj(**light_pars)
    else:
        # this condition never occurs if the __approx_dict__ is properly written
        raise AttributeError("Unable to define method of type {} for waveform generator. Check bajes.obs.kn.lightcurve.__approx_dict__".format(light_pars['type']))

    return light_func

#
# lightcurve object
#

class Lightcurve(object):
    """
        Lightcurve object
    """

    def __init__(self, times, lambdas, approx, **kwargs):
        """
            Initialize the Lightcurve with a frequency axis and the name of the approximant
        """

        self.times      = times
        self.lambdas    = lambdas
        self.approx     = approx
        logger.info("Setting {} lightcurve ...".format(self.approx))

        # get waveform generator from string
        self.light_func = __get_lightcurve_generator__(self.approx, self.times, self.lambdas, **kwargs)

    def compute_mag(self, params):

        if 'cos_iota' in params.keys():
            params['iota'] = np.arccos(params['cos_iota'])
        elif 'iota' in params.keys():
            params['cos_iota'] = np.cos(params['iota'])
        else:
            raise KeyError("Unable to read inclination parameter, information is missing. Please use iota or cosi.")

        # include band information in params
        params['photometric-lambdas'] = self.lambdas
        return self.light_func(self.times, params)

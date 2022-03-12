from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.special import i0e

from .. import erase_init_wrapper
from ...inf.likelihood import Likelihood

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

# GRAVITATIONAL-WAVE LIKELIHOOD
# Gaussian Likelihood function:
# -0.5 (d-h|d-h) = Re(d|h) - 0.5 (d|d) - 0.5 (h|h)

class GWLikelihood(Likelihood):
    """
        Log-likelihood object,
        it assumes that the data are evaluated on the same frequency axis (given as input)
    """

    def __init__(self, ifos, datas, dets, noises,
                 freqs, srate, seglen, approx,
                 nspcal=0, spcal_freqs=None,
                 nweights=0, len_weights=None,
                 marg_phi_ref=False, marg_time_shift=False,
                 **kwargs):

        # run standard initialization
        super(GWLikelihood, self).__init__()

        # set data properties
        self.ifos   = ifos
        self.dets   = dets

        # store information
        self.nspcal = nspcal
        self.nweights = nweights

        # set marginalization flags
        self.marg_phi_ref       = marg_phi_ref
        self.marg_time_shift    = marg_time_shift

        n_freqs     = None
        f_min_check = None
        f_max_check = None

        # iterate over detectors
        for ifo in self.ifos:

            self.dets[ifo].store_measurement(datas[ifo], noises[ifo],
                                             nspcal, spcal_freqs,
                                             nweights, len_weights)

            if f_min_check == None:
                f_min_check = datas[ifo].f_min
            else:
                if datas[ifo].f_min != f_min_check:
                    logger.error("Input f_min of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input f_min of data and model do not match in detector {}.".format(ifo))

            if f_max_check == None:
                f_max_check = datas[ifo].f_max
            else:
                if datas[ifo].f_max != f_max_check:
                    logger.error("Input f_max of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input f_max of data and model do not match in detector {}.".format(ifo))

            if n_freqs == None:
                n_freqs = len(datas[ifo].freqs)
            else:
                if len(datas[ifo].freqs) != n_freqs:
                    logger.error("Number of data samples does not match in detector {}.".format(ifo))
                    raise ValueError("Number of data samples does not match in detector {}.".format(ifo))

            if datas[ifo].seglen != seglen:
                logger.error("Input seglen of data and model do not match in detector {}.".format(ifo))
                raise ValueError("Input seglen of data and model do not match in detector {}.".format(ifo))

            self.logZ_noise += -0.5 * self.dets[ifo]._dd
            self.Nfr        = n_freqs
            mask            = datas[ifo].mask

        # initialize waveform generator
        from ...obs.gw.waveform import Waveform
        self.wave   = erase_init_wrapper(Waveform(freqs[mask], srate , seglen, approx))

#    def inner_prods(self, params):
#
#        # compute waveform
#        hphc    = np.array(self.wave.compute_hphc(params))
#
#        hh = 0.
#        dd = 0.
#
#        if self.marg_time_shift:
#
#            dh_arr = np.zeros(self.Nfr, dtype=complex)
#
#            for ifo in self.ifos:
#                dh_arr_thisifo, hh_thisifo, dd_thisifo = self.dets[ifo].compute_inner_products(hphc, params, self.wave.domain)
#                dh_arr = dh_arr + np.fft.fft(dh_arr_thisifo)
#                hh += hh_thisifo
#                dd += dd_thisifo
#
#            if self.marg_phi_ref:
#                dh  = np.max(np.abs(dh_arr))
#            else:
#                dh  = np.max(np.real(dh_arr))
#
#        else:
#
#            dh = 0.+0.j
#
#            # compute inner products
#            for ifo in self.ifos:
#                dh_arr_thisifo, hh_thisifo, dd_thisifo = self.dets[ifo].compute_inner_products(hphc, params, self.wave.domain)
#                dh += np.sum(dh_arr_thisifo)
#                hh += hh_thisifo
#                dd += dd_thisifo
#
#            # evaluate logL
#            if self.marg_phi_ref:
#                dh  = np.abs(dh)
#            else:
#                dh  = np.real(dh)
#
#        return dh, hh, dd

    def log_like(self, params):
        """
            log-likelihood function
        """
        # compute waveform
        logger.debug("Generating waveform for {}".format(params))
        wave    = self.wave.compute_hphc(params)
        logger.debug("Waveform generated".format(params))

        # if hp, hc == [None], [None]
        # the requested parameters are unphysical
        # Then, return -inf
        if not any(wave.plus):
            return -np.inf

        if(np.any(np.isnan(wave.plus)) or np.any(np.isnan(wave.cross))): 
            logger.warning('Nans in the waveform, with the configuration: {}. Returning -inf in the likelihood.'.format(params))
            return -np.inf
        if(np.any(np.isinf(wave.plus)) or np.any(np.isinf(wave.cross))):
            logger.warning('Infinities in the waveform, with the configuration: {}. Returning -inf in the likelihood.'.format(params))
            return -np.inf

        hh = 0.
        dd = 0.
        _psd_fact = 0.

        if self.marg_time_shift:

            dh_arr = np.zeros(self.Nfr, dtype=complex)

            # compute inner products
            for ifo in self.ifos:
                logger.debug("Projecting over {}".format(ifo))
                dh_arr_thisifo, hh_thisifo, dd_thisifo, _psdf = self.dets[ifo].compute_inner_products(wave, params, self.wave.domain, psd_weight_factor=True)
                dh_arr = dh_arr + np.fft.fft(dh_arr_thisifo)
                hh += np.real(hh_thisifo)
                dd += np.real(dd_thisifo)
                _psd_fact += _psdf

            # evaluate logL
            logger.debug("Estimating likelihood")
            if self.marg_phi_ref:
                abs_dh  = np.abs(dh_arr)
                I0_dh   = np.log(i0e(abs_dh)) + abs_dh
                R       = logsumexp(I0_dh-np.log(self.Nfr))
            else:
                re_dh   = np.real(dh_arr)
                R       = logsumexp(re_dh-np.log(self.Nfr))

        else:

            dh = 0.+0.j

            # compute inner products
            for ifo in self.ifos:
                logger.debug("Projecting over {}".format(ifo))
                dh_arr_thisifo, hh_thisifo, dd_thisifo, _psdf = self.dets[ifo].compute_inner_products(wave, params, self.wave.domain, psd_weight_factor=True)
                dh += (dh_arr_thisifo).sum()
                hh += np.real(hh_thisifo)
                dd += np.real(dd_thisifo)
                _psd_fact += _psdf

            # evaluate logL
            logger.debug("Estimating likelihood")
            if self.marg_phi_ref:
                dh  = np.abs(dh)
                R   = np.log(i0e(dh)) + dh
            else:
                R   = np.real(dh)

        logL =  -0.5*(hh + dd) + R - self.logZ_noise - 0.5*_psd_fact

        return logL

# KILO-NOVA LIKELIHOOD
# Gaussian Likelihood function:
# -0.5 (|d-L|/s)**2
class KNLikelihood(Likelihood):

    def __init__(self, comps, filters,
                 priors, prior_grid=900, kind='linear',
                 v_min=1.e-7, n_v=400,
                 n_time=400, t_start=1., t_scale='linear',
                 **kwargs):

        # run standard initialization
        super(KNLikelihood, self).__init__()

        # set data properties
        self.filters = filters

        # compute data normalization
        self.logZ_noise = -0.5*sum([np.power(self.filters.magnitudes[bi]/self.filters.mag_stdev[bi],2.) for bi in self.filters.bands()])

        # initilize time axis for lightcurve model
        if t_start > 86400:
            logger.warning("Initial time for lightcurve evaluation is larger than a day (86400 s). Setting t_start to 1h")
            t_start = 3600

        # the time axis passed to the lightcurve goes from t_start (~0) to the size of the measurement times
        # subsequently (line 489) the time axis is rescaled such that t=0 goes to t_gps
        t_size = np.max(filters.all_times)- np.min(filters.all_times)
        if 'time_shift' in priors.names:
            ip = priors.names.index('time_shift')
            t_size += priors.bounds[ip][1]-priors.bounds[ip][0]

        if t_scale=='linear':
            t_axis  = np.linspace(t_start, t_size+t_start, n_time)
        elif t_scale=='log':
            t_axis  = np.logspace(np.log10(t_start), np.log10(t_size+t_start), num=n_time)
        elif t_scale=='mixed':
            t1      = np.logspace(np.log10(t_start), np.log10(t_size+t_start), num=n_time//2)
            dt      = t_size/(2+n_time/2)
            t2      = np.linspace(t_start+dt, t_size+t_start-dt, n_time//2)
            t_axis  = np.sort(np.concatenate(t1,t2))
        else:
            raise ValueError("Unknown property {} for t_scale variable during KNLikelihood initialization.".format(t_scale))

        # initialize lightcurve model
        from ...obs.kn.lightcurve import Lightcurve
        self.light  = Lightcurve(comps, t_axis, filters.lambdas, v_min, n_v)

    def log_like(self, x):

        # compute lightcurve
        mags    = self.light.compute_mag(params)

        logL = 0.

        for bi in self.filters.bands():
            interp_mag  = np.interp(self.filters.times[bi], self.light.times+params['t_gps'], mags[bi])
            residuals   = ((self.filters.magnitudes[bi]-interp_mag)/self.filters.mag_stdev[bi])**2.
            logL       += -0.5*(residuals).sum()

        return logL

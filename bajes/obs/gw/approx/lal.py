from __future__ import division, unicode_literals, absolute_import
import numpy as np

try:
    import lalsimulation as lalsim
except Exception:
    pass

class lal_wrapper(object):

    def __init__(self, approx, domain):
    
        self.approx = lalsim.__dict__[approx]
        self.domain = domain

    def __call__(self, freqs, params):

        if self.domain == 'time' :
            return generate_timedomain_waveform(self.approx, params)
        elif self.domain == 'freq' :
            fr, hp, hc  = generate_freqdomain_waveform(self.approx, params)
            indxs       = np.where((fr>=params['f_min'])&(fr<=params['f_max']))
            return hp[indxs], hc[indxs]
        else:
            raise ValueError("Unable to generate LAL waveform, invalid domain.")


def generate_timedomain_waveform(approx, params):
    """
        SimInspiralChooseTDWaveform:
        REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
        REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
        const REAL8 m1,                             /**< mass of companion 1 (kg) */
        const REAL8 m2,                             /**< mass of companion 2 (kg) */
        const REAL8 S1x,                            /**< x-component of the dimensionless spin of object 1 */
        const REAL8 S1y,                            /**< y-component of the dimensionless spin of object 1 */
        const REAL8 S1z,                            /**< z-component of the dimensionless spin of object 1 */
        const REAL8 S2x,                            /**< x-component of the dimensionless spin of object 2 */
        const REAL8 S2y,                            /**< y-component of the dimensionless spin of object 2 */
        const REAL8 S2z,                            /**< z-component of the dimensionless spin of object 2 */
        const REAL8 distance,                       /**< distance of source (m) */
        const REAL8 inclination,                    /**< inclination of source (rad) */
        const REAL8 phiRef,                         /**< reference orbital phase (rad) */
        const REAL8 longAscNodes,                   /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
        const REAL8 eccentricity,                   /**< eccentrocity at reference epoch */
        const REAL8 UNUSED meanPerAno,              /**< mean anomaly of periastron */
        const REAL8 deltaT,                         /**< sampling interval (s) */
        const REAL8 f_min,                          /**< starting GW frequency (Hz) */
        REAL8 f_ref,                                /**< reference GW frequency (Hz) */
        LALDict *LALparams,                         /**< LAL dictionary containing accessory parameters */
        const Approximant approximant               /**< post-Newtonian approximant to use for waveform production */
    """
    LALDict = lalsim.lal.CreateDict()
    if params['lambda1'] != 0. :
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(LALDict, params['lambda1'])
    if params['lambda2'] != 0. :
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(LALDict, params['lambda2'])
    
    hp,hc = lalsim.SimInspiralChooseTDWaveform(lalsim.lal.MSUN_SI*params['mtot']*params['q']/(1.+params['q']),
                                               lalsim.lal.MSUN_SI*params['mtot']/(1.+params['q']),
                                               params['s1x'],params['s1y'],params['s1z'],
                                               params['s2x'],params['s2y'],params['s2z'],
                                               params['distance']*1e6*lalsim.lal.PC_SI,
                                               params['iota'],
                                               params['phi_ref'],
                                               0.0, params['eccentricity'], 0.0,
                                               1./params['srate'],
                                               params['f_min'],
                                               params['f_min'],
                                               LALDict,
                                               approx)
    hp = hp.data.data
    hc = hc.data.data
    return np.array(hp) , np.array(hc)

def generate_freqdomain_waveform(approx, params):
    """
        SimInspiralChooseFDWaveform:
        COMPLEX16FrequencySeries **hptilde,     /**< FD plus polarization */
        COMPLEX16FrequencySeries **hctilde,     /**< FD cross polarization */
        const REAL8 m1,                         /**< mass of companion 1 (kg) */
        const REAL8 m2,                         /**< mass of companion 2 (kg) */
        const REAL8 S1x,                        /**< x-component of the dimensionless spin of object 1 */
        const REAL8 S1y,                        /**< y-component of the dimensionless spin of object 1 */
        const REAL8 S1z,                        /**< z-component of the dimensionless spin of object 1 */
        const REAL8 S2x,                        /**< x-component of the dimensionless spin of object 2 */
        const REAL8 S2y,                        /**< y-component of the dimensionless spin of object 2 */
        const REAL8 S2z,                        /**< z-component of the dimensionless spin of object 2 */
        const REAL8 distance,                   /**< distance of source (m) */
        const REAL8 inclination,                /**< inclination of source (rad) */
        const REAL8 phiRef,                     /**< reference orbital phase (rad) */
        const REAL8 longAscNodes,               /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
        const REAL8 eccentricity,               /**< eccentricity at reference epoch */
        const REAL8 UNUSED meanPerAno,          /**< mean anomaly of periastron */
        // frequency sampling parameters, no default value
        const REAL8 deltaF,                     /**< sampling interval (Hz) */
        const REAL8 f_min,                      /**< starting GW frequency (Hz) */
        const REAL8 f_max,                      /**< ending GW frequency (Hz) */
        REAL8 f_ref,                            /**< Reference frequency (Hz) */
        LALDict *LALparams,                     /**< LAL dictionary containing accessory parameters */
        const Approximant approximant           /**< post-Newtonian approximant to use for waveform production */
    """
    LALDict = lalsim.lal.CreateDict()
    if params['lambda1'] != 0. :
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(LALDict, params['lambda1'])
    if params['lambda2'] != 0. :
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(LALDict, params['lambda2'])

    hp,hc = lalsim.SimInspiralChooseFDWaveform(lalsim.lal.MSUN_SI*params['mtot']*params['q']/(1.+params['q']),
                                               lalsim.lal.MSUN_SI*params['mtot']/(1.+params['q']),
                                               params['s1x'],params['s1y'],params['s1z'],
                                               params['s2x'],params['s2y'],params['s2z'],
                                               params['distance']*1e6*lalsim.lal.PC_SI,
                                               params['iota'],
                                               params['phi_ref'],
                                               0.0, params['eccentricity'], 0.0,
                                               1./params['seglen'],
                                               params['f_min'],
                                               params['f_max'],
                                               params['f_min'],
                                               LALDict,
                                               approx)
    hp      = hp.data.data
    hc      = hc.data.data
    L       = len(hp)
    freq    = np.arange(L)/params['seglen']
    return freq, np.array(hp, dtype=complex) , np.array(hc, dtype=complex)

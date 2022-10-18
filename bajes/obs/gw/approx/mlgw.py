from __future__ import division, unicode_literals, absolute_import
import numpy as np

try:
    import mlgw.GW_generator as generator
except ImportError:
    pass

try:
    from mlgw_bns import Model, ParametersWithExtrinsic
except ImportError:
    pass

__bns_pars__ = ['mass_ratio', 'lambda_1', 'lambda_2', 'chi_1', 'chi_2',
                'distance_mpc', 'inclination', 'total_mass',
                'reference_phase']
__bjs_pars__ = ['q', 'lambda1', 'lambda2', 's1z', 's2z',
                'distance', 'iota', 'mtot',
                'phi_ref']

def params_bajes_to_mlgwbns(pars):
    # by convention all waveform are aligned at the center of the segment
    # Then, frequency-domain waveform are genereted with the convetion time_shift=0 (according to LALSim)
    # NOTE: In bajes/obs/gw/waveform.py we shift of seglen/2  every frequency-domain waveform
    out = {ki : pars[kj] for ki,kj in zip(__bns_pars__, __bjs_pars__)}
    out['time_shift'] = 0.
    return out

class mlgw_bns_wrapper():

    """
        Class wrapper for MLGW-BNS waveform
    """

    def __init__(self, **kwargs):

        self.model = Model.default()

    def __call__(self, freqs, params):
        bns_params      = ParametersWithExtrinsic(**params_bajes_to_mlgwbns(params))
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        return hp_i, hc_i

class mlgw_bns_nrpmw_wrapper():

    """
        Class wrapper for MLGW-BNS-NRPMw waveform
    """

    def __init__(self, **kwargs):

        self.model = Model.default()

        from .nrpmw import nrpmw_attach_wrapper
        self.nrpmw_func = nrpmw_attach_wrapper

    def __call__(self, freqs, params):
        bns_params      = ParametersWithExtrinsic(**params_bajes_to_mlgwbns(params))
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        hp_p, hc_p      = self.nrpmw_func(freqs, params)
        return hp_i+hp_p, hc_i+hc_p

class mlgw_bns_nrpmw_recal_wrapper():

    """
        Class wrapper for MLGW-BNS-NRPMw waveform with recalibration
    """

    def __init__(self, **kwargs):

        self.model = Model.default()

        from .nrpmw import nrpmw_attach_recal_wrapper
        self.nrpmw_func = nrpmw_attach_recal_wrapper

    def __call__(self, freqs, params):
        bns_params      = ParametersWithExtrinsic(**params_bajes_to_mlgwbns(params))
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        hp_p, hc_p      = self.nrpmw_func(freqs, params)
        return hp_i, hc_i

class mlgw_bns_nrpmw_wrapper():

    """
        Class wrapper for MLGW-BNS-NRPMw waveform
    """

    def __init__(self, freqs, seglen, srate):

        self.model = Model.default()

        self.freqs  = freqs
        self.srate  = srate
        self.seglen = seglen

        from .nrpmw import nrpmw_attach_wrapper
        self.nrpmw_func = nrpmw_attach_wrapper

    def __call__(self, freqs, params):
        bns_params      = ParametersWithExtrinsic(**params_bajes_to_mlgwbns(params))
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        hp_p, hc_p      = self.nrpmw_func(freqs, params)
        return hp_i+hp_p, hc_i+hc_p

class mlgw_bns_nrpmw_recal_wrapper():

    """
        Class wrapper for MLGW-BNS-NRPMw waveform with recalibration
    """

    def __init__(self, freqs, seglen, srate):

        self.model = Model.default()

        self.freqs  = freqs
        self.srate  = srate
        self.seglen = seglen

        from .nrpmw import nrpmw_attach_recal_wrapper
        self.nrpmw_func = nrpmw_attach_recal_wrapper

    def __call__(self, freqs, params):
        bns_params      = ParametersWithExtrinsic(**params_bajes_to_mlgwbns(params))
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        hp_p, hc_p      = self.nrpmw_func(freqs, params)
        return hp_i+hp_p, hc_i+hc_p

class mlgw_wrapper(object):

    """
        Class wrapper for MLGW waveform
        folder = 0 -> TEOBResumS (w/o NQC)
    """

    def __init__(self, seglen, srate, **kwargs):

        self.generator  = generator.GW_generator()
        self.srate      = srate
        self.seglen     = seglen
        self.dt         = 1./self.srate
        self.times      = np.arange(int(self.srate*self.seglen))*self.dt - self.seglen/2.

    def __call__(self, freqs, params):
        return self.td_waveform(params)

    def td_waveform(self, params):

        #theta = [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
        theta = [params['mtot']*params['q']/(1.+params['q']),
                 params['mtot']/(1.+params['q']),
                 params['s1z'],
                 params['s2z'],
                 params['distance'],
                 params['iota'],
                 params['phi_ref']]

        hp, hc = self.generator.get_WF(theta, self.times)
        if len(hp) == 1:
            return np.array(hp[0]), np.array(hc[0])
        else :
            return np.array(hp), np.array(hc)

class mlteobnqc_wrapper(object):

    """
        Class wrapper for MLGW waveform
        folder = 1 -> TEOBResumS+NQC
    """


    def __init__(self, seglen, srate, **kwargs):

        self.generator  = generator.GW_generator(folder=1)
        self.srate      = srate
        self.seglen     = seglen
        self.dt         = 1./self.srate
        self.times      = np.arange(int(self.srate*self.seglen))*self.dt - self.seglen/2.

    def __call__(self, freqs, params):
        return self.td_waveform(params)

    def td_waveform(self, params):

        #theta = [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
        theta = [params['mtot']*params['q']/(1.+params['q']),
                 params['mtot']/(1.+params['q']),
                 params['s1z'],
                 params['s2z'],
                 params['distance'],
                 params['iota'],
                 params['phi_ref']]

        hp, hc = self.generator.get_WF(theta, self.times)
        if len(hp) == 1:
            return np.array(hp[0]), np.array(hc[0])
        else :
            return np.array(hp), np.array(hc)

class mlseobv4_wrapper(object):

    """
        Class wrapper for MLGW waveform
        folder = 3 -> SEOBNRv4
    """

    def __init__(self, seglen, srate, **kwargs):

        self.generator  = generator.GW_generator(folder=3)
        self.srate      = srate
        self.seglen     = seglen
        self.dt         = 1./self.srate
        self.times      = np.arange(int(self.srate*self.seglen))*self.dt - self.seglen/2.

    def __call__(self, freqs, params):
        return self.td_waveform(params)

    def td_waveform(self, params):

        #theta = [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
        theta = [params['mtot']*params['q']/(1.+params['q']),
                 params['mtot']/(1.+params['q']),
                 params['s1z'],
                 params['s2z'],
                 params['distance'],
                 params['iota'],
                 params['phi_ref']]

        hp, hc = self.generator.get_WF(theta, self.times)
        if len(hp) == 1:
            return np.array(hp[0]), np.array(hc[0])
        else :
            return np.array(hp), np.array(hc)

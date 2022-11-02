from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging, warnings
logger = logging.getLogger(__name__)

from ..utils import compute_lambda_tilde, compute_delta_lambda

__url__ = 'https://pypi.org/project/mlgw/'
try:
    import mlgw.GW_generator as generator
except ImportError:
    warnings.warn("Unable to import MLGW package. Please see related documentation at: {}".format(__url__))
    logger.warning("Unable to import MLGW package. Please see related documentation at: {}".format(__url__))
    pass

__url__ = 'https://pypi.org/project/mlgw-bns/'
try:
    from mlgw_bns import Model, ParametersWithExtrinsic
except ImportError:
    warnings.warn("Unable to import MLGW-BNS package. Please see related documentation at: {}".format(__url__))
    logger.warning("Unable to import MLGW-BNS package. Please see related documentation at: {}".format(__url__))
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

def mlgw_bns_merger_time_shift(mt, q, lambda1, lambda2, chi1, chi2):
    """
        Estimate delay from coalescence time to reference time.

        Currently, MLGW-BNS (<= 0.12.1) does not include information on
        the reference time of coalescence. Then, this moment is estimated
        with calibrated relation. The calibration sets includes random samples in the ranges:
            1. M = 2.8 M⊙ ,  q = 1 ,       χ1, χ2 = 0 ,            Λ1, Λ2 ∈ [5, 5000] ;
            2. M = 2.8 M⊙ ,  q ∈ [1,2] ,   χ1, χ2 = 0 ,            Λ1, Λ2 ∈ [5, 5000].
            3. M = 2.8 M⊙ ,  q ∈ [1,2] ,   χ1, χ2 ∈ [-0.2, 0.2] ,  Λ1, Λ2 ∈ [5, 5000].

        The residual time-shifts are obtained with following script:

        from mlgw_bns.data_management import Residuals, ParameterRanges
        from mlgw_bns import Model
        import numpy as np

        p_ranges = ParameterRanges(mass_range       = (2.0, 4.0),
                                   q_range          = (1.0, 2.0),
                                   lambda1_range    = (5.0, 5000.0),
                                   lambda2_range    = (5.0, 5000.0),
                                   chi1_range       = (-0.2, 0.2),
                                   chi2_range       = (-0.2, 0.2))

        m = Model(parameter_ranges=p_ranges)

        _, training_params, training_residuals = m.dataset.generate_residuals(5_000, flatten_phase=False)

        training_timeshifts = training_residuals.flatten_phase(m.dataset.frequencies_hz)

        np.save('parameters.npy', training_params.parameter_array)
        np.save('timeshifts.npy', training_timeshifts)
    """

    m1              = mt*q/(1+q)
    m2              = mt/(1+q)
    nu              = m1*m2/mt/mt
    X               = 1-4*nu
    sqX             = np.sqrt(X)
    lt              = compute_lambda_tilde(m1,m2,lambda1,lambda2)
    dl              = compute_delta_lambda(m1,m2,lambda1,lambda2)
    ld              = lambda2-lambda1
    ls              = lambda1+lambda2
    ce              = (m1*chi1 + m2*chi2)/mt

    cs = {    'a0': 0.0020759621961942603,
              'a1': -0.003070536961079697,
              'a2': -9.629458339368572e-07,
              'a3': 0.000542447368295823,
              'a4': 6.604879000806726e-09,
              'b0': -6.585504061714901e-12,
              'b1': 5.652710843348325e-13,
              'bx': -2.293953484226671e-14,
              'c0': 12.882556914189799,
              'c1': 11.21937846139295,
              'c2': 11.449609847910668,
              'c3': -2.9681147636094796,
              'd0': -0.07143247198447567,
              'd1': -0.9359464525773223,
              'b2': 0.002444447726984135,
              'b3': -3.8982563658704605e-05,
              'b4': -0.00017640077699324582,
              'e0': 0.1812511561795588,
              'e1': -1.2949443130142897,
              'e2': -1.6835328420336901,
              'e3': -0.46264137662511656,
              'b5': -0.010454941422013362,
              'b6': 2.838006604314393,
              'b7': -0.00033645778450926226,
              'b8': -0.010507369168043812}

    lt_fit      = lt + cs['d0']*sqX*ld + cs['d1']*sqX*dl
    fx_q1_s0    = cs['a0']*(1.+cs['a1']*lt_fit + cs['a2']*lt_fit**2)/(1.+cs['a3']*lt_fit + cs['a4']*lt_fit**2)
    X_corr      = (1+cs['c0']*X+cs['c1']*X*X)/(1+cs['c2']*X+cs['c3']*X*X)
    S_corr      = (1+cs['e0']*ce + cs['e1']*ce**2 + cs['e2']*X*ce)/(1+cs['e3']*ce)**2

    correct1    = cs['b0']*ld**2 + cs['b1']*ls**2 + cs['bx']*ls*ld
    correct2    = cs['b2']*X *(1. + cs['b3']*ld  + cs['b4']*ls)
    correct3    = b5*ce*(1 + b6*X + b7*lt) + b8*X*(chi1-chi2)

    return fx_q1_s0 * X_corr * S_corr + correct1 + correct2 + correct3

    dt_m28          = fx + corr

    return dt_m28*mt/2.8

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
        dt_merg         = mlgw_bns_merger_time_shift(params['mtot'],    params['q'],
                                                     params['lambda1'], params['lambda2'],
                                                     params['s1z'],     params['s2z'])
        time_shift_f    = np.exp(2j*np.pi*freqs*dt_merg)
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        hp_p, hc_p      = self.nrpmw_func(freqs, params)
        return hp_i*time_shift_f+hp_p, hc_i*time_shift_f+hc_p

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
        dt_merg         = mlgw_bns_merger_time_shift(params['mtot'],    params['q'],
                                                     params['lambda1'], params['lambda2'],
                                                     params['s1z'],     params['s2z'])
        time_shift_f    = np.exp(2j*np.pi*freqs*dt_merg)
        hp_i, hc_i      = self.model.predict(freqs, bns_params)
        hp_p, hc_p      = self.nrpmw_func(freqs, params)
        return hp_i*time_shift_f+hp_p, hc_i*time_shift_f+hc_p

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

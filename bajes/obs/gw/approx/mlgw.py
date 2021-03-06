from __future__ import division, unicode_literals, absolute_import
import numpy as np

try:
    import mlgw.GW_generator as generator
except Exception:
    pass

class mlgw_wrapper(object):
    
    """
        Class wrapper for MLGW waveform
        folder = 0 -> TEOBResumS (w/o NQC)
    """

    def __init__(self, seglen, srate):

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


    def __init__(self, seglen, srate):

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

    def __init__(self, seglen, srate):

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




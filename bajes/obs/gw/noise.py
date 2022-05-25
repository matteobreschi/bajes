from __future__ import division, unicode_literals, absolute_import
import numpy as np

from .strain import Series, filtering

def evaluate_psd(strain, dt, subseglen, overlap_fraction = 0.99):
    """
        Compute the power spectral density by
        Welch's average periodogram method.
        The vector strain is divided into subsegments
        with subseglen duration.

        Arguments:

            - strain            : noise data strain array
            - dt                : time step of the input strain
            - subseglen         : seglen of subsegment
            - overlap_fraction  : fraction of overlapping elements, default 0.99

        Return:
            - freqs : frequency axis
            - psd   : power spectral density
    """

    from scipy.signal import welch

    seglen      = int(len(strain)*dt)
    srate       = int(1./dt)
    nperseg     = int(subseglen * srate)
    noverlap    = int(overlap_fraction*nperseg)
    window      = np.blackman(nperseg)

    freqs, psd = welch(strain,
                       fs              = 1./dt,
                       window          = window,
                       nperseg         = nperseg,
                       noverlap        = noverlap,
                       return_onesided = True,
                       scaling         = 'density')

    return freqs, psd

def read_asd_from_txt(path):
    return np.genfromtxt(path, usecols=[0,1], unpack=True)

def get_design_sensitivity(ifo):

    import os
    from ... import __path__
    main_path = os.path.abspath(__path__[0]+'/pipe/data/gw/asd/design/')

    if ifo=='L1' or ifo=='H1' or ifo=='I1':
        filename = 'LIGO-P1200087-v18-aLIGO_DESIGN.txt'
    elif ifo=='V1':
        filename = 'LIGO-P1200087-v18-AdV_DESIGN.txt'
    elif ifo=='K1':
        filename = 'LIGO-T1600593-v1-KAGRA_Design.txt'
    elif 'ET' in ifo:
        filename = 'ET-0001A-18_ETD.txt'
    elif ifo=='CE':
        #TODO remove or rename this, since the `CE-T2000017-v5-40km.txt` is the official new main CE noise curve
        filename = 'LIGO-P1600143-v18-CE.txt'
    # Please find the description of the 'CE-40', 'CE-40-lf', 'CE-20', 'CE-20-pm' can be found at bajes/pipe/data/gw/asd/design/README.md
    elif ifo in ['CE-South-lf', 'CE-North-lf']:
        filename = 'CE-T2000017-v5-40km_lf.txt'
    elif ifo in ['CE-South-40', 'CE-North-40']:
        filename = 'CE-T2000017-v5-40km.txt'
    elif ifo in ['CE-South-pm', 'CE-North-pm']:
        filename = 'CE-T2000017-v5-20km_pm.txt'
    elif ifo in ['CE-South-20', 'CE-North-20']:
        filename = 'CE-T2000017-v5-20km.txt'
    else:
        raise AttributeError("Design ASD not available for requested detector. Design ASD is available for the following IFOs: H1, L1, V1, K1, I1, ET, CE.")
    asd_path = os.path.join(main_path , filename)
    return np.genfromtxt(asd_path , usecols=[0,1], unpack=True)

def get_event_sensitivity(ifo, event):

    import os
    from ... import __path__
    asd_path        = os.path.abspath(__path__[0]+'/pipe/data/gw/asd/events/{}/{}_ASD.txt'.format(event, ifo))

    return np.genfromtxt(asd_path , usecols=[0,1], unpack=True)

def get_event_calibration(ifo, event):

    import os
    from ... import __path__
    spcal_path      = os.path.abspath(__path__[0]+'/pipe/data/gw/spcal/events/{}/{}_CalEnv.txt'.format(event, ifo))

    f, mag, phi, sigma_mag_low, sigma_phi_low, sigma_mag_up, sigma_phi_up  = np.transpose(np.genfromtxt(spcal_path , usecols=[0,1,2,3,4,5,6]))
    amp_sigma = (sigma_mag_up - sigma_mag_low)/2.
    phi_sigma = (sigma_phi_up - sigma_phi_low)/2.
    return [f,amp_sigma,phi_sigma]

class Noise(object):
    """
        Gaussian and stationary noise
    """

    def __init__(self, freqs, asd, f_min=None, f_max=None, filter=False):

        if len(freqs) != len(asd):
            logger.error("Numbers of data points do not match. Please check the input ASD data.")
            raise RuntimeError("Numbers of data points do not match. Please check the input ASD data.")

        # check order
        _i = np.argsort(freqs)

        from scipy.interpolate import interp1d

        self.freqs = freqs[_i]
        self.df    = np.median(np.diff(self.freqs))

        if f_max == None:
            self.f_max = np.max(self.freqs)
        else:
            self.f_max = f_max

        if f_min == None:
            self.f_min = np.min(self.freqs)
        else:
            self.f_min = f_min

        if filter == True:
            self.power_spectrum     = filtering(self.freqs, asd[_i]*asd[_i], [self.f_min,self.f_max], type='bandpass', order=4)
            self.amp_spectrum       = np.sqrt(self.power_spectrum)
        else:
            self.amp_spectrum       = asd[_i]
            self.power_spectrum     = self.amp_spectrum*self.amp_spectrum

        assert len(self.amp_spectrum) ==  len(self.freqs)

        self.freqs_pad      = self.freqs
        self.amp_spec_pad   = self.amp_spectrum
        self.pow_spec_pad   = self.power_spectrum

        # enlong ASD/PSD to f=0 to avoid problems during interpolation
        if self.freqs[0] > self.df:
            l_pad               = int(round(np.min(self.freqs)/self.df))+1
            self.freqs_pad      = np.append(np.flip(np.min(self.freqs) - np.arange(1,l_pad+1)*self.df), self.freqs)
            self.amp_spec_pad   = np.append(np.ones(l_pad)*self.amp_spectrum[0], self.amp_spectrum)
            self.pow_spec_pad   = np.append(np.ones(l_pad)*self.power_spectrum[0], self.power_spectrum)

        # enlong ASD/PSD above f_max if needed
        if self.f_max*2 > np.max(self.freqs_pad):
            l_pad               = int(round((self.f_max*2 - np.max(self.freqs_pad))/self.df))+1
            self.freqs_pad      = np.append(self.freqs_pad, np.max(self.freqs_pad)+np.arange(1,l_pad+1)*self.df)
            self.amp_spec_pad   = np.append(self.amp_spec_pad, np.ones(l_pad)*self.amp_spectrum[-1])
            self.pow_spec_pad   = np.append(self.pow_spec_pad,np.ones(l_pad)*self.power_spectrum[-1])

        assert len(self.amp_spec_pad) ==  len(self.freqs_pad)
        assert len(self.pow_spec_pad) ==  len(self.freqs_pad)

        # obs: linear interpolation is performed,
        # because higher orders give ASD/PSD < 0 (and this may screw up everything)
        self.psd_interp_func        = interp1d(self.freqs,     self.power_spectrum, fill_value=np.inf, bounds_error=False)
        self.asd_interp_func        = interp1d(self.freqs,     self.amp_spectrum,   fill_value=np.inf, bounds_error=False)
        self.asd_interp_func_pad    = interp1d(self.freqs_pad, self.amp_spec_pad,   fill_value=np.inf, bounds_error=False)
        self.psd_interp_func_pad    = interp1d(self.freqs_pad, self.pow_spec_pad,   fill_value=np.inf, bounds_error=False)

    def interp_psd(self, fr):
        asd = self.asd_interp_func(fr)
        return asd * asd

    def interp_asd(self, fr):
        return self.asd_interp_func(fr)

    def interp_asd_pad(self, fr):
        return self.asd_interp_func_pad(fr)

    def interp_psd_pad(self, fr):
        return self.psd_interp_func_pad(fr)

    # to be updated, see generate_noise_gauss.py
    def generate_fake_noise(self, seglen, srate=4096., t_gps=1126259462., filter=False):
        """
            Generate fake gaussian noise from ASD.

            Arguments:
                - seglen  : period of the output segment
                - srate   : sampling rate of the output segment (optional, default 4096)
                - t_gps   : GPS time, center of the time series (optional, default 1126259462, i.e. GW150914)

            Return:
                - fake_series : numpy.array conatining the time series of the artificial noise strain

        """

        dt      = 1./srate
        f_max   = srate/2.
        df      = 1./seglen
        N_FD    = int(round(seglen*srate//2+1))
        fr_out  = np.arange(N_FD)*df

        # interpolate PSD
        psd = self.interp_psd_pad(fr_out)

        # # ensure continuity outside freuency bounds
        # psd[np.where(fr_out<self.f_min)] = psd[np.max(np.where(fr_out<=self.f_min))]
        # psd[np.where(fr_out>self.f_max)] = psd[np.min(np.where(fr_out>=self.f_max))]


        # filter PSD
        if filter:
            # ensure continuity outside freuency bounds
            psd[np.where(fr_out<self.f_min)] = 0.0
            psd[np.where(fr_out>self.f_max)] = 0.0
            # psd = filtering(fr_out, psd, [self.f_min,self.f_max], type='bandpass', order=4)

        sigma   = 0.5*np.sqrt(psd/df)

        # generate (two) noise series
        noise1  = 1j * np.random.normal(np.zeros(len(sigma)), sigma)
        noise1  += np.random.normal(np.zeros(len(sigma))    , sigma)

        import matplotlib.pyplot as plt

        series1 = Series('freq' , noise1 , srate=srate, seglen=seglen , importfreqs= fr_out,
                         f_min=self.f_min, f_max=f_max, t_gps=t_gps, alpha_taper=0., filter=True)

        noise2  = 1j * np.random.normal(np.zeros(len(sigma)), sigma)
        noise2  += np.random.normal(np.zeros(len(sigma))    , sigma)
        series2 = Series('freq' , noise2 , srate=srate, seglen=seglen , importfreqs= fr_out,
                         f_min=self.f_min, f_max=f_max, t_gps=t_gps, alpha_taper=0., filter=True)

        # feather together noise1 & noise2,
        # in order to avoid unwanted specularities, repetitions, etc
        x       = np.cos(np.pi*np.arange(len(series1.time_series))/len(series1.time_series))
        y       = np.sin(np.pi*np.arange(len(series2.time_series))/len(series2.time_series))
        noise   = series1.time_series*x + series2.time_series*y

        return noise

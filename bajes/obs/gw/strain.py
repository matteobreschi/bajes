from __future__ import division, unicode_literals, absolute_import
import numpy as np
from scipy.signal import tukey

import logging
logger = logging.getLogger(__name__)

def next_power_of_2(x):
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))

def fft_doublesided(h, dt):
    """
        Compute the double-sided FFT of a given time series
        -----------
        h = array, time series
        We evaluate the FFT of this function
        dt = float
        Timestep of h
        -----------
        freq = array
        frequency array on which tha FFT is evaluated
        hfft = array, complex
        FFT of h
    """
    N       = len(h)
    hfft    = np.fft.fft(h) * dt
    f       = np.fft.fftfreq(N, d=dt)
    return f , hfft

def fft(h, dt):
    """
        Compute the single-sided FFT of a given time series,
        the output is NOT multiplied by a factor 2
        -----------
        h = array, time series
        We evaluate the FFT of this function
        dt = float
        Timestep of h
        -----------
        freq = array
        frequency array on which tha FFT is evaluated
        hfft = array, complex
        FFT of h
    """
    N    = len(h)
    hfft = np.fft.rfft(h) * dt
    f    = get_freq_ax(N,dt)
    return f , hfft

def ifft(u , srate, seglen, t0=0.):
    """ Compute the inverse FFT of a given complex frequency series,
        this function assumes that the input is one-sided FFT
        -----------
        u = array, frequency series
        We evaluate the inverse FFT of this function
        srate = float
        Sampling rate (= 1/dt) in Hertz
        seglen = float
        Duration of the segment in seconds
        t0 = float
        Central time value
        -----------
        time = array
        time array on which tha FFT is evaluated, starting fro t0
        hifft = array, float
        inverse FFT of u
        """
    N   = int(srate*seglen)
    s   = np.fft.irfft(u,N)
    t   = get_time_ax(N,srate,t0)
    return t , s*srate

def get_freq_ax(N,dt,fmin=0.):
    """ Return the frequency axis of a
        frequency-domain series
        __________
        N    : number of output points *
        dt   : time step
        fmin : initial frequency. Default 0 Hz
    """
    f = np.fft.rfftfreq(N, d=dt)
    if fmin == 0.:
        return f
    else:
        inds = np.where(f>=fmin)
        return f[inds]

def get_time_ax(N,srate,tgps=0.):
    """ Return the time axis of a
        time-domain series
        __________
        N    : number of output points
        fNyq : Nyquist frequency
        tgps : central time value
    """
    dt      = 1./srate
    seglen  = N*dt
    time    = np.arange(N)*dt + tgps - seglen/2.
    return time

def lagging(h,n):
    if n > 0:
        return np.concatenate((np.full(n, 0.), h[:-n]))
    elif n < 0:
        return np.concatenate((h[-n:], np.full(-n, 0.)))
    else:
        return h

def padding(h, dt, where='center', padlen=None):
    """ Perform zero-padding on a given strain (time-domain)
        __________
        h      : strain to be padded
        dt     : time step
        where  : position of the padding,
                 i.e. 'bottom', 'center' or 'top'
                 or the value for the requested central indexs.
                 Default 'center'
        padlen : length of the padding, if none it rounds up to the closer (integer) multiple of 2
    """
    if padlen == None:
        Tin = dt * len(h)
        padlen = int(np.ceil((next_power_of_2(Tin) - Tin)/dt))

    if isinstance(where,str):
        if where == 'bottom':
            return np.append(np.zeros(padlen), h)
        elif where == 'center':
            q = 0
            if padlen%2 == 1: q = 1
            return np.append(np.zeros(padlen//2) ,np.append(h,np.zeros(padlen//2+q)))
        elif where == 'top':
            return np.append(h,np.zeros(padlen))

    elif isinstance(where,int):
        lenFin      = len(h) + padlen
        lenFin_top  = lenFin//2
        lenFin_bot  = lenFin - lenFin_top
        Nbelow      = int(np.ceil(np.max([lenFin_bot - where,0])))
        Nabove      = int(np.ceil(np.max([lenFin_top - len(h) + where,0])))
        return np.append(np.zeros(Nbelow),np.append(h,np.zeros(Nabove)))

    else:
        raise ValueError("Invalid location variable (where) for padding. Please use 'bottom', 'center', 'top' or the value for the requested central index.")

def windowing(h, alpha=0.1):
    """ Perform windowing with Tukey window on a given strain (time-domain)
        __________
        h     : strain to be tapered
        alpha : Tukey filter slope parameter. Suggested value: alpha = 1/4/seglen
    """
    window  = tukey(len(h), alpha)
    wfact   = np.mean(window**2)
    return h*window, wfact

def filtering(fr, hfft, f_cut, type='lowpass', order=4):
    """ Perform frequency-domain filtering with 4th order Butterworth filter on a given strain (frequency-domain)
        __________
        fr    : frequency axis, used to evaluate the position of the filter
        hfft  : strain to be filtered
        f_cut : cut-off frequency. 2-dim array in case of 'bandpass'
        type  : type of filter to apply, i.e. 'lowpass', 'highpass', 'bandpass'
    """
    from scipy.signal import butter, freqs
    b, a = butter(order, f_cut, type, analog=True)
    w, h = freqs(b, a)

    filter = np.interp(fr , w, np.abs(h))
    return hfft * filter

def bandpassing(h, srate, f_min, f_max, order=4):
    """ Perform time-domain filtering with 4th order Butterworth filter on a given strain (time-domain)
        __________
        hfft  : strain to be filtered
        srate : sampling rate
        f_min : lower cut-off frequency
        f_max : upper cut-off frequency
        order : filter order (default 4)
        """
    from scipy.signal import butter, filtfilt
    bb, ab = butter(order, [f_min*2./srate, f_max*2./srate], btype='bandpass')
    hbp = filtfilt(bb, ab, h, method="gust")
    return hbp

def lowpassing(h, srate, f_min, order=4):
    """ Perform time-domain filtering with 4th order Butterworth filter on a given strain (time-domain)
        __________
        hfft  : strain to be filtered
        srate : sampling rate
        f_cut : lower cut-off frequency
        order : filter order (default 4)
        """
    from scipy.signal import butter, filtfilt
    bb, ab = butter(order, f_min*2./srate, btype='lowpass')
    hbp = filtfilt(bb, ab, h, method="gust")
    return hbp

def highpassing(h, srate, f_max, order=4):
    """ Perform time-domain filtering with 4th order Butterworth filter on a given strain (time-domain)
        __________
        hfft  : strain to be filtered
        srate : sampling rate
        f_cut : upper cut-off frequency
        order : filter order (default 4)
        """
    from scipy.signal import butter, filtfilt
    bb, ab = butter(order, f_max*2./srate, btype='highpass')
    hbp = filtfilt(bb, ab, h, method="gust")
    return hbp

class Series(object):
    """ A strain series in time/frequency domain
    """

    def __init__(self               ,
                 type               ,
                 series             ,
                 srate              ,
                 seglen      = None ,
                 f_min       = None ,
                 f_max       = None ,
                 t_gps       = 0.   ,
                 only        = False,
                 filter      = False,
                 alpha_taper = None ,
                 importfreqs = []
                ):

        """ Initialize series
            ___________
            type    : string, specify the type of input series, i.e. 'time' or 'freq'
            series  : real or complex array, input time (real) or frequency (complex) series
                    - for time series, t_gps is the central time value of the array
                    - for freq series, f_min is the cutoff freq of the high-pass filter
                    (we assumes that the freq series starts from f=0 Hz)
            srate   : float, sampling rate of the given series [Hz]
            seglen  : float (optional), length of the given series [s].
                    If None, internally compute the current seglens.
                    If given seglen differs from actual seglen, the series is truncated or padded
            f_min   : float (optional), minimum frequency of the input series [Hz]. Default None
            f_max   : float (optional), maximum frequency of the input series [Hz]. Default None
            t_gps   : float (optional), reference time. Defaut 0s
            only    : bool (optional), if True, store only input series, without computing fft/ifft. Default False.
            filter  : bool (optional), if True, apply butterworth filter. Default False
            alpha_taper : float (optional), alpha parameter of the Tukey window. Default 0.4/seglen
            importfreqs : array (optional), in the case of a given freq-series, it is possible to pass manually the frequency axis
        """

        self.window_factor = 1.

        if type == 'time':
            self.srate      = srate
            self.dt         = 1./self.srate
            self.f_min      = f_min
            self.f_max      = f_max
            self.t_gps      = t_gps
            self.f_Nyq      = self.srate/2.

            # temporary values
            raw_N           = len(series)

            if self.f_min == None:
                self.f_min = 0.
            if self.f_max == None:
                self.f_max = self.f_Nyq

            # When no seglen is provided, read the time series, window it and pad it to the next power of two. Set seglen using the full lenght of the padded data.
            if seglen == None:

                seglen_tmp = len(series) * self.dt

                if alpha_taper == None:
                    self.alpha_taper = 0.4/seglen_tmp
                else:
                    self.alpha_taper = alpha_taper

                wind_series, wfact = windowing(series,self.alpha_taper)
                #Using padding, increase the series seglen up to the next power of two.
                self.time_series   = padding(wind_series,self.dt,'center')
                self.seglen        = len(self.time_series) * self.dt

            else:

                self.seglen = seglen
                finalN      = int(np.ceil(self.seglen*self.srate))

                if alpha_taper == None:
                    self.alpha_taper = 0.4/self.seglen
                else:
                    self.alpha_taper = alpha_taper

                if finalN%2 == 1 :
                    finalN += 1

                if raw_N == finalN:
                    wind_series, wfact  = windowing(series,self.alpha_taper)
                    self.time_series    = wind_series

                elif finalN > raw_N:
                    logger.warning("Input seglen for time series is greater than the total data-length. The time series will be padded to get the requested input.")
                    padlen              = finalN - raw_N
                    wind_series, w_tmp  = windowing(series,self.alpha_taper)
                    wfact               = np.mean(np.append(tukey(len(series), self.alpha_taper)**2.,
                                                            np.zeros(padlen))) # compute corrected window factor
                    self.time_series    = padding(wind_series,self.dt,'center',padlen)

                elif finalN < raw_N:
                    logger.warning("Input seglen for time series is smaller than the total data-length. The series will be truncated to get the requested input.")
                    Nhalf = int(len(series)//2)
                    wind_series, wfact  = windowing(series[Nhalf-finalN//2:Nhalf+finalN//2],self.alpha_taper)
                    self.time_series    = wind_series

            self.window_factor  = wfact

            if filter:
                self.time_series    = bandpassing(self.time_series , self.srate, self.f_min , self.f_max)

            self.df                 = 1/self.seglen
            self.times              = get_time_ax(len(self.time_series), self.srate, self.t_gps)

            if only:
                self.freqs = None
                self.freq_series = None
            else:
                self.freqs, self.freq_series = fft(self.time_series, self.dt)

        elif type == 'freq':

            self.srate  = srate
            self.dt     = 1./self.srate
            self.f_Nyq  = self.srate/2.
            raw_N       = len(series)

            if seglen == None:
                logger.warning("Frequency series defined without seglen. Seglen will be estimated assuming f_max = f_Nyq and f_min = 0")
                self.seglen = self.f_Nyq/raw_N
            else:
                self.seglen = seglen

            self.df          = 1./self.seglen
            self.dt          = 1./self.srate
            self.t_gps       = t_gps

            self.f_min       = f_min
            self.f_max       = f_max
            if self.f_min == None:
                self.f_min  = 0.
            if self.f_max == None:
                self.f_max  = self.f_Nyq

            # read freq series + band-pass filtering between [f_min,f_Nyq]
            if len(importfreqs) == 0 :
                self.freqs  = get_freq_ax((len(series)-1)*2, self.dt, self.f_min)
            else:
                self.freqs  = importfreqs

            if filter:
                self.freq_series    = filtering(self.freqs, series, [self.f_min,self.f_max] , type='bandpass')
            else:
                self.freq_series    = series

            assert len(self.freqs) == len(self.freq_series)

            if only:
                self.times = None
                self.time_series = None
            else:
                # compute ifft
                # TODO : include check for frequency axis
                self.times, self.time_series    = ifft(self.freq_series, self.srate, self.seglen, self.t_gps)

        else:
            logger.error("Type of series not specified or wrong. Please use 'time' or 'freq'.")
            raise AttributeError("Type of series not specified or wrong. Please use 'time' or 'freq'.")

        if isinstance(self.freqs, np.ndarray):
            self.mask = np.where((self.freqs>=self.f_min)&(self.freqs<=self.f_max))
        else:
            self.freqs = np.array(self.freqs)
            self.mask = np.where((self.freqs>=self.f_min)&(self.freqs<=self.f_max))

    def __add__(self,   other):
        return self.freq_series + other.freq_series

    def __sub__(self,   other):
        return self.freq_series - other.freq_series

    def __prod__(self,  other):
        return np.conj(self.freq_series) * other.freq_series

    def bandpassing(self, flow, fhigh, order=4):
        self.time_series    = bandpassing(self.time_series, self.srate, flow, fhigh, int(order))
        _, self.freq_series = fft(self.time_series, self.dt)

    def lowpassing(self, flow, order=4):
        self.time_series    = lowpassing(self.time_series, self.srate, flow, int(order))
        _, self.freq_series = fft(self.time_series, self.dt)

    def highpassing(self, fhigh, order=4):
        self.time_series    = highpassing(self.time_series, self.srate, fhigh, int(order))
        _, self.freq_series = fft(self.time_series, self.dt)

    def whitening(self, noise):

        self.freq_series                = self.freq_series/noise.interp_asd_pad(self.freqs)/np.sqrt(self.srate)
        self.times, self.time_series    = ifft(self.freq_series, self.srate, self.seglen, self.t_gps)

    def interp_freq_series(self, new_freqs):
        """ Cubic interpolation of frequency series
            ___________
            new_freqs : float array
            New frequency axis
            ___________
            interp_htilde : complex array
            Frequency series interpolated over new_freqs
        """
        from scipy.interpolate import interp1d
        phi         = np.unwrap(np.angle(self.freq_series))
        amp         = np.abs(self.freq_series)
        phi_interp  = interp1d(self.freqs, phi)
        amp_interp  = interp1d(self.freqs, phi)
        return amp_interp(new_freqs) * np.exp(1j * phi_interp(new_freqs))

    def shift_freq_series(self, dt):
        return self.freq_series * np.exp(-2j*np.pi*dt*self.freqs)

    def real_product(self, series2, psd):
        """ Real inner-product:
            Re(s|h) = 4 * Re( sum( h^* s / PSD ) )
            evaluated on the entire frequency.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
        """
        num     = np.conj(self.freq_series)*series2.freq_series
        den     = psd * self.window_factor
        return 4.*np.sum(np.real(num/den))*self.df

    def imag_product(self, series2, psd):
        """ Real inner-product:
            Im(s|h) = 4 * Im( sum( h^* s / PSD ) )
            evaluated on the entire frequency.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            """
        num     = np.conj(self.freq_series)*series2.freq_series
        return 4.*np.sum(np.imag(num/psd))*self.df

    def complex_product(self, series2, psd):
        """ Real inner-product:
            (s|h) = 4 * sum( h^* s / PSD )
            evaluated on the entire frequency.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            """
        num     = np.conj(self.freq_series)*series2.freq_series
        den     = psd * self.window_factor
        return 4.*np.sum(num/den)*self.df

    def abs_product(self, series2, psd):
        """ Abs inner-product:
            |(s|h)| = 4 * sum( | h^* s | / PSD ) )
            evaluated on the entire frequency.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            ----------
            series2  = bajes.gwmodule.waveform.Series object
            second frequency series
            psd    = float array
            power spectral density
        """
        num     = np.conj(self.freq_series)*series2.freq_series
        den     = psd * self.window_factor
        return 4.*np.sum(np.abs(num/den))*self.df

    def self_product(self, psd):
        """ Self inner-product:
            (s|s) = 4 * sum( |s|^2 / PSD )
            evaluated on the entire frequency.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            ----------
            series2  = bajes.gwmodule.waveform.Series object
            second frequency series
            psd    = float array
            power spectral density
        """
        num = np.abs(self.freq_series)**2.
        den     = psd * self.window_factor
        return 4.*np.sum(num/den)*self.df

    def inner_product(self, series2, noise, f_bounds=None):
        """ Compute inner product (complex) between the input series and an other series
            in the frequency range [f_min,f_max].
            Second series is assumed to be evaluated on the same freq axis of the primary series.
            PSD is interpolated in the frequency values of the primary series.
            ----------
            series2  = bajes.gwmodule.waveform.Series object
            second frequency series
            noise    = bajes.gwmodule.noise.Noise object
            noise spectral density
            f_bounds = None or 2d array
            [minimum frequency, maximum frequency], if None
            the inner product is evaluated on the entire self.freqs
            ----------
            return inner product
        """
        # restrict domain to [f_min,f_max]
        if f_bounds==None:
            fr      = self.freqs
            fs1     = self.freq_series
            fs2     = series2.freq_series
        else:
            mask    = np.where( (self.freqs>=f_bounds[0]) & (self.freqs<=f_bounds[1]) )
            fr      = self.freqs[mask]
            fs1     = self.freq_series[mask]
            fs2     = series2.freq_series[mask]

        # interpolate PSD and frequency series
        psd   = noise.interp_psd_pad(fr)

        num     = np.conj(fs1)*fs2
        den     = psd * self.window_factor
        return 4.*np.sum(num/den)*self.df

    def residuals(self, series2, psd):
        """ Compute residuals between the input series and an other series
            on the frequency axis of the primary series.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            The residuals are equivalent to the contribution of the
            data in the log-likelihood, i.e. logL ~ - 0.5 * residuals
            ----------
            series2  = bajes.gwmodule.waveform.Series object
            second frequency series
            psd    = float array
            power spectral density
            f_bounds = None or 2d array
            [minimum frequency, maximum frequency], if None
            the inner product is evaluated on the entire self.freqs
            ----------
            return inner product
        """
        res     = self.freq_series - series2.freq_series
        num     = np.conj(res)*res
        den     = psd * self.window_factor
        return 4.*np.sum(np.real(num/den))*self.df

    def snr(self, series2, noise, norm=None):
        """ Compute SNR between the input series and an other series,
            on the frequency axis of the primary series.
            The second series is taken as the template.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            ----------
            series2  = bajes.obs.gw.strain.Series object
            noise    = bajes.obs.gw.noise.Noise object
            ----------
            return tshift, snr_oftime
        """
        l       = len(self.freq_series)
        den     = psd * self.window_factor

        if norm == None:
            norm    = np.sqrt(series2.self_product(den))

        s1_conj             = np.zeros(l, dtype=complex)
        s1_conj[self.mask]  = np.conj(self.freq_series[self.mask])
        s2                  = np.zeros(l, dtype=complex)
        s2[self.mask]       = series2.freq_series[self.mask]

        snr_oftime  = 4.*self.df*np.real(np.fft.fft(s1_conj*s2/den))/norm
        tshift      = np.arange(len(snr_oftime))*self.dt

        return tshift, snr_oftime

    def snr_phimax(self, series2, noise, norm=None):
        """ Compute SNR (phi maximized) between the input series and an other series,
            on the frequency axis of the primary series.
            The second series is taken as the template.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            ----------
            series2  = bajes.obs.gw.strain.Series object
            noise    = bajes.obs.gw.noise.Noise object
            ----------
            return tshift, snr_oftime
        """
        l       = len(self.freq_series)
        den     = noise.interp_psd_pad(self.freqs) * self.window_factor

        if norm == None:
            norm    = np.sqrt(series2.self_product(den))

        s1_conj             = np.zeros(l, dtype=complex)
        s1_conj[self.mask]  = np.conj(self.freq_series[self.mask])
        s2                  = np.zeros(l, dtype=complex)
        s2[self.mask]       = series2.freq_series[self.mask]

        snr_oftime  = 4.*self.df*np.abs(np.fft.fft(s1_conj*s2/den))/norm
        tshift      = np.arange(len(snr_oftime))*self.dt

        return tshift, snr_oftime

    def snr_complex(self, series2, noise, norm=None):
        """ Compute SNR (phi maximized) between the input series and an other series,
            on the frequency axis of the primary series.
            The second series is taken as the template.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            ----------
            series2  = bajes.obs.gw.strain.Series object
            noise    = bajes.obs.gw.noise.Noise object
            ----------
            return tshift, snr_oftime
        """

        den     = noise.interp_psd_pad(self.freqs) * self.window_factor

        if norm == None:
            norm    = np.sqrt(series2.self_product(den))

        s1_conj             = np.zeros(l, dtype=complex)
        s1_conj[self.mask]  = np.conj(self.freq_series[self.mask])
        s2                  = np.zeros(l, dtype=complex)
        s2[self.mask]       = series2.freq_series[self.mask]

        snr_oftime  = 4.*self.df*np.fft.fft(s1_conj*s2/den)/norm
        tshift      = np.arange(len(snr_oftime))*self.dt

        return tshift, snr_oftime

    def overlap(self, series2, psd, timelen=1):
        """ Compute overlap between the input series and an other series,
            on the frequency axis of the primary series.
            PSD and second series is assumed to be evaluated on the same
            freq axis of the primary series.
            The overlap is the match (normalized inner product) fo the two series
            maximized over time delay and reference phase.
            ----------
            series2  = bajes.gwmodule.waveform.Series object
            second frequency series
            psd    = float array
            power spectral density
            timelen = float
            Length in seconds of time window. Default 1.
            ----------
            return max(overlap), (time, overlap)
        """
        nsteps  = round(timelen/self.dt)+1
        inds    = np.arange(nsteps)-nsteps//2

        norm    = np.sqrt(series2.self_product(psd)) * np.sqrt(self.self_product(psd))
        s1_conj = np.conj(self.freq_series)

        overlap = np.array([4.*self.df*np.abs(np.sum(s1_conj*series2.shift_freq_series(self.dt*i)/psd))/norm for i in inds], dtype=float)
        tshift  = inds*self.dt

        return np.max(overlap), (tshift, overlap)

from __future__ import division, unicode_literals, absolute_import
import numpy as np
from numpy import cos, sin

from .utils import tdwf_2_fdwf, fdwf_2_tdwf
from .strain import lagging
from ... import CLIGHT_SI

def gmst_accurate(gps_time):
    from astropy.time import Time
    gmst = Time(gps_time, format='gps', scale='utc',
                location=(0, 0)).sidereal_time('mean').rad
    return gmst

def compute_spcalenvs(ifo, nspcal, params):
    amp_spcal =  np.array([params['spcal_amp{}_{}'.format(i,ifo)] for i in range(nspcal)])
    phi_spcal =  np.array([params['spcal_phi{}_{}'.format(i,ifo)] for i in range(nspcal)])
    return (1.+amp_spcal)*np.exp(1j*phi_spcal)

def compute_psdweights(ifo, nweights, len_weights, params):
    return np.concatenate([np.ones(len_weights[i])*params['weight{}_{}'.format(i,ifo)] for i in range(nweights)])

# project array in time-domain
# function used for inject waveform from txt file
def calc_project_array_td(det, hp, hc, dt, ra, dec, psi, t_gps):
    """ Project given waveform on detector
        ----------
        det = Detector class
        wf  = Waveform class
        hp  = plus polarization component of the waveform
        hc  = cross polarization component of the waveform
        ra  = right ascenscion of the source
        dec = declination of the source
        psi = waveform polarization angle
        t_gps = trigger time of the event, with respect to Earth center
        ----------
        h = F+ h+ + Fx hx , projected strain
        """

    fplus , fcross  = det.antenna_pattern(ra, dec, psi, t_gps)
    time_delay      = det.time_delay_from_earth_center(ra , dec , t_gps)

    h = fplus*hp + fcross*hc
    proj_series = lagging(h, int(round(time_delay/dt)))
    return proj_series

def get_detector_information(ifo):
    """ Return information on selected detector
        ------
        ifo: str
        The two-character detector string, i.e. H1, L1, V1, K1, G1
        ______
        latitude, longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt, yarm_tilt
    """
    if ifo=='H1':
        latitude        = 0.81079526383
        longitude       = -2.08405676917
        elevation       = 142.554
        xarm_azimuth    = 2.1991043855
        yarm_azimuth    = 3.7699007123
        xarm_tilt       = -6.195e-4
        yarm_tilt       = 1.25e-5
    elif ifo=='L1':
        latitude        = 0.53342313506
        longitude       = -1.58430937078
        elevation       = -6.574
        xarm_azimuth    = 3.4508039105
        yarm_azimuth    = 5.0216002373
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='V1':
        latitude        = 0.76151183984
        longitude       = 0.18333805213
        elevation       = 51.884
        xarm_azimuth    = 1.2316334746
        yarm_azimuth    = 2.8024298014
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='K1':
        latitude        = 0.6355068497
        longitude       = 2.396441015
        elevation       = 414.181
        xarm_azimuth    = 0.5166831721
        yarm_azimuth    = 2.0874762035
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='G1':
        latitude        = 0.91184982752
        longitude       = 0.17116780435
        elevation       = 114.425
        xarm_azimuth    = 2.02358884
        yarm_azimuth    = 0.377195322
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='I1':
        latitude        = 0.2484185302005
        longitude       = 1.334013324941
        elevation       = 0.
        xarm_azimuth    = 2.1991043855
        yarm_azimuth    = 3.7699007123
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='ET1':
        latitude        = 0.76151183984
        longitude       = 0.18333805213
        elevation       = 51.884
        xarm_azimuth    = 0.33916285222
        yarm_azimuth    = 5.57515060820
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='ET2':
        latitude        = 0.76299307990
        longitude       = 0.18405858870
        elevation       = 59.735
        xarm_azimuth    = 4.52795305701
        yarm_azimuth    = 3.48075550581
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='ET3':
        latitude        = 0.76270463257
        longitude       = 0.18192996730
        elevation       = 59.727
        xarm_azimuth    = 2.43355795462
        yarm_azimuth    = 1.38636040342
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif ifo=='CE':
        latitude        = 0.81079526383
        longitude       = -2.08405676917
        elevation       = 142.554
        xarm_azimuth    = 2.1991043855
        yarm_azimuth    = 3.7699007123
        xarm_tilt       = -6.195e-4
        yarm_tilt       = 1.25e-5
    # The CE-North and CE-South detectors are --fiducial-- sites and configurations for the main US CE (in Idaho) and a secondary CE in Australia, respectively. 
    # These were used for the CE Horizon Study: https://arxiv.org/abs/2109.09882
    elif 'CE-North' in ifo:
        latitude        = 0.764918
        longitude       = -1.9691740
        elevation       = 0.
        xarm_azimuth    = -1.57079632679
        yarm_azimuth    = 0.
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    elif 'CE-South' in ifo:
        latitude        = -0.593412
        longitude       = 2.5307270
        elevation       = 0.
        xarm_azimuth    = -0.7853981634
        yarm_azimuth    =  0.7853981634
        xarm_tilt       = 0.
        yarm_tilt       = 0.
    else:
        raise ValueError("Can't get information from input detector. Please check you use a correct name. Available options: ['H1', 'L1', 'V1', 'K1', 'G1', 'I1', 'ET1', 'ET2', 'ET3', 'CE'].")

    return latitude, longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt, yarm_tilt

class Detector(object):
    """
        A gravitational wave ground-based interferometer
    """
    def __init__(self, detector_init, t_gps=None):
        """
            Initialize object representing a gravitational-wave detector

            Arguments:
                - detector_init: str or list or dict
                    if string , the two-character detector string, i.e. H1, L1, V1, K1, G1, I1, CE, ET
                    if list   , specify the following 7 elements: latitude, longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt, yarm_tilt
                    if dict   , specify the same previous 7 elements as keyword arguments
                - t_gps: float, Earth's rotation will be estimated from a reference time.

        """

        # initialize detector properties
        if isinstance(detector_init, str):
            self.ifo = detector_init
            self.latitude, self.longitude, self.elevation, self.xarm_azimuth, self.yarm_azimuth, self.xarm_tilt, self.yarm_tilt = get_detector_information(self.ifo)
        elif isinstance(detector_init, list):
            self.ifo = 'UNKNOWN'
            self.latitude, self.longitude, self.elevation, self.xarm_azimuth, self.yarm_azimuth, self.xarm_tilt, self.yarm_tilt = detector_init
        elif isinstance(detector_init, dict):
            self.ifo = 'UNKNOWN'
            for ki in list(detector_init.keys()):
                self.__dict__[ki] = detector_init[ki]
        else:
            raise RuntimeError("Unable to read initialization argument for Detector object. Please use a compatible string, a list or a dictionary with the required information.")

        # compute detector geometry
        self.x_arm      = self.compute_arm(self.xarm_tilt, self.xarm_azimuth)
        self.y_arm      = self.compute_arm(self.yarm_tilt, self.yarm_azimuth)
        self.location   = self.compute_location()
        self.response   = self.compute_response()

        # initialize times
        self.reference_time = t_gps
        self.sday = None
        self.gmst_reference = None
        self.compute_gmst_reference()

    def compute_response(self):
        return 0.5 * (np.einsum('i,j->ij', self.x_arm, self.x_arm) - np.einsum('i,j->ij', self.y_arm, self.y_arm))

    def compute_location(self):
        # define semi-major and semi-minor axes of Earth [m]
        semi_major_axis = 6378137
        semi_minor_axis = 6356752.314
        radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(self.latitude)**2 + semi_minor_axis**2 * np.sin(self.latitude)**2)**(-0.5)
        x_comp = (radius + self.elevation) * np.cos(self.latitude) * np.cos(self.longitude)
        y_comp = (radius + self.elevation) * np.cos(self.latitude) * np.sin(self.longitude)
        z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + self.elevation) * np.sin(self.latitude)
        return np.array([x_comp, y_comp, z_comp])

    def compute_arm(self, arm_tilt, arm_azimuth):
        e_long  = np.array([-np.sin(self.longitude), np.cos(self.longitude), 0])
        e_lat   = np.array([-np.sin(self.latitude) * np.cos(self.longitude), -np.sin(self.latitude) * np.sin(self.longitude), np.cos(self.latitude)])
        e_h     = np.array([np.cos(self.latitude) * np.cos(self.longitude), np.cos(self.latitude) * np.sin(self.longitude), np.sin(self.latitude)])
        return (np.cos(arm_tilt) * np.cos(arm_azimuth) * e_long + np.cos(arm_tilt) * np.sin(arm_azimuth) * e_lat + np.sin(arm_tilt) * e_h)

    def compute_gmst_reference(self):
        if self.reference_time is not None:
            from astropy.units.si import sday
            self.sday = float(sday.si.scale)
            self.gmst_reference = gmst_accurate(self.reference_time)
        else:
            raise RuntimeError("Can't get accurate sidereal time without GPS reference time!")

    def gmst_estimate(self, gps_time):
        if self.reference_time is None:
            return gmst_accurate(gps_time)
        if self.gmst_reference is None:
            self.compute_gmst_reference()
        dphase = (gps_time - self.reference_time) / self.sday * (2.0 * np.pi)
        gmst = (self.gmst_reference + dphase) % (2.0 * np.pi)
        return gmst

    def light_travel_time_to_detector(self, det):
        """
            Return the light travel time from this detector

            Arguments:
                - det: bajes.obs.gw.Detector

            Return:
                - time: float, the light travel time in seconds
        """
        d = self.location - det.location
        return float(d.dot(d)**0.5 / CLIGHT_SI)

    def antenna_pattern(self, right_ascension, declination, polarization, t_gps):
        """
            Detector response.
            Inspired from XLALComputeDetAMResponse and PyCBC.detector

            Arguments:
                - right_ascension   : float
                - declination       : float
                - polarization      : float

            Return:
                - fplus: float, the plus polarization factor for this sky location
                - fcross: float, the cross polarization factor for this sky location
        """

        # t_gps is measured from the center of the Earth,
        # then we have to take into account the time delay

        geometric_delay = self.time_delay_from_earth_center(right_ascension, declination, t_gps)
        t_gps   += geometric_delay
        gha     = self.gmst_estimate(t_gps) - right_ascension

        cosgha = cos(gha)
        singha = sin(gha)
        cosdec = cos(declination)
        sindec = sin(declination)
        cospsi = cos(polarization)
        sinpsi = sin(polarization)

        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 =  sinpsi * cosdec
        x = np.array([x0, x1, x2])
        dx = self.response.dot(x)

        y0 =  sinpsi * singha - cospsi * cosgha * sindec
        y1 =  sinpsi * cosgha + cospsi * singha * sindec
        y2 =  cospsi * cosdec
        y = np.array([y0, y1, y2])
        dy = self.response.dot(y)

        if hasattr(dx, 'shape'):
            fplus = (x * dx - y * dy).sum(axis=0)
            fcross = (x * dy + y * dx).sum(axis=0)
        else:
            fplus = (x * dx - y * dy).sum()
            fcross = (x * dy + y * dx).sum()

        return fplus, fcross

    def time_delay_from_earth_center(self, right_ascension, declination, t_gps):
        """
            Return the time delay from the earth center

            Arguments:
                - right_ascension   : float
                - declination       : float

            Return:
                - time_delay : float
        """
        return self.time_delay_from_location(np.array([0, 0, 0]),
                                             right_ascension,
                                             declination,
                                             t_gps)

    def time_delay_from_location(self, other_location, right_ascension, declination, t_gps):
        """
            Return the time delay from the given location to detector for
            a signal with the given sky location
            In other words return `t1 - t2` where `t1` is the
            arrival time in this detector and `t2` is the arrival time in the
            other location.

            Arguments:
                - other_location    : np.array (3-dim), coordinates of other location
                - right_ascension   : float
                - declination       : float
                - t_gps             : float, the GPS time (in s) of the signal.
        """
        ra_angle = self.gmst_estimate(t_gps) - right_ascension
        cosd = cos(declination)

        e0 = cosd * cos(ra_angle)
        e1 = cosd * -sin(ra_angle)
        e2 = sin(declination)

        ehat    = np.array([e0, e1, e2])
        dx      = other_location - self.location
        return dx.dot(ehat) / CLIGHT_SI

    def time_delay_from_detector(self, other_detector, right_ascension, declination, t_gps):
        """
            Return the time delay from the given to detector for a signal with
            the given sky location; i.e. return `t1 - t2` where `t1` is the
            arrival time in this detector and `t2` is the arrival time in the
            other detector. Note that this would return the same value as
            `time_delay_from_earth_center` if `other_detector` was geocentric.

            Arguments:
                - other_detector    : bajes.obs.gw.Detector
                - right_ascension   : float
                - declination       : float
                - t_gps             : float, the GPS time (in s) of the signal.
            """
        return self.time_delay_from_location(other_detector.location,
                                             right_ascension,
                                             declination,
                                             t_gps)

    def optimal_orientation(self, t_gps):
        """
            Return the optimal orientation in right ascension and declination
            for a given GPS time.

            Arguments:
                - t_gps             : float, the GPS time (in s) of the signal.

            Return:
                - right ascension   : float
                - declination       : float
        """
        ra  = self.longitude+(self.gmst_estimate(t_gps)%(2.*np.pi))
        dec = self.latitude
        return ra, dec

    def project_fdwave(self, wave, params, tag):
        """
            Project waveform on Detector, with frequency-domain output

            Arguments:
                - wave      : bajes.obs.gw.waveform.PolarizationTuple
                - params    : dict, parameters of the waveform, reuired: ra, dec, psi, t_gsp, time_shift
                - tag       : Domain of the waveform, time or freq

            Return:
                - projected wave : np.array, waveform projected on this detector in the frequency-domain
        """
        # compute F+,Fx for the detecor at the moment t-gps + time-shift
        fplus , fcross  = self.antenna_pattern(params['ra'], params['dec'], params['psi'], params['t_gps']+params['time_shift'])
        # compute delay from Earth geocenter to detector
        delay   = self.time_delay_from_earth_center(params['ra'], params['dec'], params['t_gps']+params['time_shift']) + params['time_shift']
        # apply antenna patterns
        proj_h  = fplus*wave.plus + fcross*wave.cross

        if tag == 'freq':
            return proj_h*np.exp(-2j*np.pi*self.freqs*delay)

        elif tag == 'time':
            # tdwf_2_fdwf (compute fft, interpolate) + apply time delay
            return tdwf_2_fdwf(self.freqs, proj_h, 1./self.srate) * np.exp(-2j*np.pi*self.freqs*delay)

    def project_tdwave(self, wave, params, tag):
        """
            Project waveform on Detector, with time-domain output

            Arguments:
                - wave      : bajes.obs.gw.waveform.PolarizationTuple
                - params    : dict, parameters of the waveform, reuired: ra, dec, psi, t_gsp, time_shift
                - tag       : Domain of the waveform, time or freq

            Return:
                - projected wave : np.array, waveform projected on this detector in the time-domain
        """
        # compute F+,Fx for the detecor at the moment t-gps + time-shift
        fplus , fcross  = self.antenna_pattern(params['ra'], params['dec'], params['psi'], params['t_gps']+params['time_shift'])
        # compute delay from Earth geocenter to detector
        delay = self.time_delay_from_earth_center(params['ra'] , params['dec'] , params['t_gps']+params['time_shift']) + params['time_shift']
        # apply antenna patterns
        proj_h  = fplus*wave.plus + fcross*wave.cross

        if tag == 'time':
            # apply time delay from geocenter
            return lagging(proj_h, int(round(delay*self.srate)))

        elif tag == 'freq':
            # compute ifft and apply time delay from geocenter
            proj_wave = fdwf_2_tdwf(self.freqs, proj_h * np.exp(-2j*np.pi*self.freqs*delay), 1./self.srate)
            return proj_wave

    def store_measurement(self,
                          series,
                          noise,
                          nspcal      = 0,
                          spcal_freqs = None,
                          nweights    = 0,
                          len_weights = None):
        """
            Store observation in Detector

            Arguments:
                - series        : bajes.obs.gw.Series
                - noise         : bajes.obs.gw.Noise
                - nspcal        : int, number of sp. cal. nodes
                - spcal_freqs   : np.array, sp. cal. frequency
                - nweights      : int, number of PSD weights
                - len_weights   : np.array, len of each weight
        """

        # set & check data properties
        assert self.reference_time == series.t_gps
        self.srate  = series.srate
        self.seglen = series.seglen

        # get data mask
        self._mask = series.mask
        self._nfr  = len(series.freqs)

        # get data
        self.freqs  = np.copy(series.freqs[series.mask])
        self.data   = np.copy(series.freq_series[series.mask])
        self.psd    = noise.interp_psd_pad(self.freqs)*series.window_factor

        # set calibration envelopes
        self.nspcal      = nspcal
        self.spcal_freqs = spcal_freqs

        # set PSD weights
        self.nweights    = nweights
        self.len_weights = len_weights

        self._dd = (4./self.seglen) * np.sum(np.abs(self.data)**2./self.psd)

    def compute_inner_products(self, hphc, params, tag, psd_weight_factor=False):
        """
            Compute inner products

            Arguments:
                - hphc              : bajes.obs.gw.waveform.PolarizationTuple
                - params            : dict, parameters
                - tag               : str, waveform domain
                - psd_weight_factor : bool, return PSD weight likelihood factor

            Return:
                - d_h           : numpy.array, corresponding to the integrand of the inner product
                - h_h           : float, self inner product of the waveform
                - d_d           : float, self inner product of the data
                - psd_factor    : float
        """
        # project waveform on detector
        wav = self.project_fdwave(hphc, params, tag)
        psd = self.psd

        # apply calibration envelopes
        if self.nspcal > 0:
            cal = compute_spcalenvs(self.ifo, self.nspcal, params)
            wav = wav*np.interp(self.freqs, self.spcal_freqs, cal)

        # apply PSD weights and compute (d|d)
        if self.nweights == 0:
            dd  = self._dd
            _w  = 0.
        else:
            weights = compute_psdweights(self.ifo, self.nweights, self.len_weights, params)
            _w  = (np.log(weights)).sum()
            psd = psd * weights
            dd  = (4./self.seglen) * (np.abs(self.data)**2./psd).sum()

        hh = (4./self.seglen) * (np.abs(wav)**2./psd).sum()
        dh = np.zeros(self._nfr, dtype=complex)
        dh[self._mask] = (4./self.seglen) * np.conj(self.data)*wav/psd

        if psd_weight_factor:
            return dh, hh, dd, _w
        else:
            return dh, hh, dd

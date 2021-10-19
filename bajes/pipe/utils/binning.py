"""
    Functions and methods inspired by GWBinning,
    https://bitbucket.org/dailiang8/gwbinning,
    arXiv:1806.08792v2 [astro-ph.IM] (2018),
    B. Zackay et al.
"""
from __future__ import division, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.integrate import trapz
from scipy.special import i0e

from ..likelihood import Likelihood
from ...utils import erase_init_wrapper, list_2_dict
from ...obs.gw.detector import compute_spcalenvs


# construct frequency bins for relative binning
def setup_bins(f_full, f_lo, f_hi, chi=1., eps=0.5):
    """
        construct frequency binning
        f_full: full frequency grid
        [f_lo, f_hi] is the frequency range one would like to use for matched filtering
        chi, eps are tunable parameters [see Barak, Dai & Venumadhav 2018]
        return the number of bins, a list of bin-edge frequencies, and their positions in the full frequency grid
    """
    from scipy.interpolate import interp1d

    f = np.linspace(f_lo, f_hi, 10000)
    # f^ga power law index
    ga = np.array([-5.0/3.0, -2.0/3.0, 1.0, 5.0/3.0, 7.0/3.0])
    dalp = chi*2.0*np.pi/np.absolute(f_lo**ga - f_hi**ga)
    dphi = np.sum(np.array([ np.sign(ga[i])*dalp[i]*f**ga[i] for i in range(len(ga)) ]), axis=0)
    Dphi = dphi - dphi[0]
    # now construct frequency bins
    Nbin = int(Dphi[-1]//eps)
    Dphi2f = interp1d(Dphi, f, kind='slinear', bounds_error=False, fill_value=0.0)
    Dphi_grid = np.linspace(Dphi[0], Dphi[-1], Nbin+1)
    # frequency grid points
    fbin = Dphi2f(Dphi_grid)
    # indices of frequency grid points in the FFT array
    fbin_ind = np.array([ np.argmin(np.absolute(f_full - ff)) for ff in fbin ])
    # make sure grid points are precise
    fbin = np.array([ f_full[i] for i in fbin_ind ])

    return (Nbin, fbin, fbin_ind)

# compute summary data given a bin partition and fiducial waveforms
def compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0):
    """
        Compute summary data
        Need to compute for each detector
        Parameters:
        f is the full frequency grid (regular grid; length = n_sample/2 + 1)
        fbin is the bin edges
        fbin_ind gives the positions of bin edges in the full grid
        ndtct is the number of detectors
        psd is a list of PSDs
        sFT is a list of frequency-domain strain data
        h0  is a list of fiducial waveforms
        Note that sFT and h0 need to be provided with the full frequency resolution
        """
    # total number of frequency bins
    Nbin = len(fbin) - 1
    # total duration of time-domain sequence
    T = 1.0/np.median(np.diff(f))

    # arrays to store summary data
    sdat_A0 = []
    sdat_A1 = []
    sdat_B0 = []
    sdat_B1 = []

    # loop over detectors
    for k in range(ndtct):

        a0 = np.array([ (4.0/T)*np.sum(sFT[k][fbin_ind[i]:fbin_ind[i+1]]
                                     *np.conjugate(h0[k][fbin_ind[i]:fbin_ind[i+1]])
                                     /psd[k][fbin_ind[i]:fbin_ind[i+1]]) for i in range(Nbin)])

        b0 = np.array([ (4.0/T)*np.sum(np.absolute(h0[k][fbin_ind[i]:fbin_ind[i+1]])**2
                                     /psd[k][fbin_ind[i]:fbin_ind[i+1]]) for i in range(Nbin)])

        a1 = np.array([ (4.0/T)*np.sum(sFT[k][fbin_ind[i]:fbin_ind[i+1]]
                                     *np.conjugate(h0[k][fbin_ind[i]:fbin_ind[i+1]])
                                     /psd[k][fbin_ind[i]:fbin_ind[i+1]]
                                     *(f[fbin_ind[i]:fbin_ind[i+1]] - 0.5*(fbin[i] + fbin[i+1]))) for i in range(Nbin)])

        b1 = np.array([ (4.0/T)*np.sum((np.absolute(h0[k][fbin_ind[i]:fbin_ind[i+1]])**2)
                                     /psd[k][fbin_ind[i]:fbin_ind[i+1]]
                                     *(f[fbin_ind[i]:fbin_ind[i+1]] - 0.5*(fbin[i] + fbin[i+1]))) for i in range(Nbin)])

        sdat_A0.append(a0)
        sdat_A1.append(a1)
        sdat_B0.append(b0)
        sdat_B1.append(b1)

    return np.array([sdat_A0, sdat_A1, sdat_B0, sdat_B1])


# Gaussian Likelihood function -0.5 * (s-h|s-h) with Frequency binning
class GWBinningLikelihood(Likelihood):
    """
        Log-likelihood object,
        it assumes that the data are evaluated on the same frequency axis (given as input)
    """

    def __init__(self,
                 ifos, datas, dets, noises, fiducial_params,
                 freqs, srate, seglen, approx,
                 nspcal=0, spcal_freqs=None,
                 nweights=0, len_weights=None,
                 marg_phi_ref=False, marg_time_shift=False,
                 **kwargs):

        # run standard initialization
        super(GWBinningLikelihood, self).__init__()

        # set dictionaries of bajes objects
        self.ifos   = ifos
        self.dets   = {ifo : erase_init_wrapper(dets[ifo]) for ifo in self.ifos}

        self.srate  = srate
        self.seglen = seglen

        # frequency binning needs f=0, but this value is value is unphysical.
        # some fiducial template might give an error,
        # then we add a little bit to it
        if freqs[0] == 0.:
            freqs[0] += freqs[1]/100.

        # generate h0
        from ...obs.gw.waveform import Waveform
        i_wav       = np.where(freqs>=fiducial_params['f_min'])
        wave0       = Waveform(freqs[i_wav], self.srate , self.seglen, approx)
        h0p, h0c    = wave0.compute_hphc(fiducial_params)

        # fill below f_min
        l_low   = len(freqs) - len(np.concatenate(i_wav))
        h0p     = np.append(np.zeros(l_low), h0p)
        h0c     = np.append(np.zeros(l_low), h0c)

        psds   = []
        sFTs   = []
        h0s    = []

        self.dd_nopsdweights = {}
        self.dd = 0.

        f_min_check = None
        f_max_check = None

        for i,ifo in enumerate(self.ifos):

            self.dets[ifo].store_measurement(datas[ifo], noises[ifo])

            # store initial data and noise, needed in line 177
            self.dets[ifo].store_measurement(datas[ifo], noises[ifo], nspcal=nspcal, spcal_freqs=spcal_freqs)

            # check if frequency ranges are consistent between data and model for every IFO
            if f_min_check == None:
                f_min_check = datas[ifo].f_min
            else:
                if datas[ifo].f_min != f_min_check:
                    logger.error("Input f_min of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input parameter (f_min) of data and model do not match in detector {}.".format(ifo))

            if f_max_check == None:
                f_max_check = datas[ifo].f_max
            else:
                if datas[ifo].f_max != f_max_check:
                    logger.error("Input f_Nyq of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input parameter (f_Nyq) of data and model do not match in detector {}.".format(ifo))

            if datas[ifo].seglen != self.seglen:
                logger.error("Input seglen of data and model do not match in detector {}.".format(ifo))
                raise ValueError("Input parameter (seglen) of data and model do not match in detector {}.".format(ifo))

            # compute PSDs and quantities for logL evaluation
            psds.append(noises[ifo].interp_psd_pad(freqs)*datas[ifo].window_factor)
            sFTs.append(datas[ifo].freq_series)
            h0s.append(self.dets[ifo].project_fdwave([h0p,h0c], fiducial_params, wave0.domain))

            self.dd_nopsdweights[ifo] = datas[ifo].inner_product(datas[ifo],noises[ifo],[datas[ifo].f_min,datas[ifo].f_max])
            self.dd += np.real(self.dd_nopsdweights[ifo])

        f_min = f_min_check
        f_max = f_max_check

        # store relative binning data
        self.Nbin, self.fbin, self.fbin_ind = setup_bins(freqs, f_min, f_max)
        self.sdat   = compute_sdat(freqs, self.fbin, self.fbin_ind, len(self.ifos), psds, sFTs, h0s)
        self.h0_bin = np.array([h0s[i][self.fbin_ind] for i,ifo in enumerate(self.ifos)])
        self.fax    = np.array([0.5*(self.fbin[i]+self.fbin[i+1]) for i in range(self.Nbin)])

        # initialize waveform generator
        self.wave   = erase_init_wrapper(Waveform(self.fbin, self.srate , self.seglen, approx))

        # modify detector frequency axis for projection
        for ifo in self.ifos:
            self.dets[ifo].data     = None
            self.dets[ifo].psd      = None
            self.dets[ifo].freqs    = self.fbin
            self.dets[ifo].srate    = self.srate
            self.dets[ifo].seglen   = self.seglen

        # set calibration envelopes
        self.nspcal     = nspcal
        if self.nspcal > 0.:
            self.spcal_freqs = spcal_freqs

        # set marginalization flags
        self.marg_phi_ref = marg_phi_ref

    def inner_products_singleifo(self, i, ifo, params, hphc):

        wav = self.dets[ifo].project_fdwave(hphc, params, self.wave.domain)

        if self.nspcal > 0:
            cal = compute_spcalenvs(ifo, self.nspcal, params)
            cal = np.interp(self.fbin, self.spcal_freqs, cal)
            wav = wav*cal

        rf      = self.compute_rf(wav, i)
        dh, hh  = self.prods_sdat(i, rf)
        return dh, hh

    def inner_prods(self, params):

        #generate waveform
        hphc    = np.array(self.wave.compute_hphc(params))

        # dh , hh
        inner_prods = np.transpose([list(self.inner_products_singleifo(i, ifo, params, hphc)) for i,ifo in enumerate(self.ifos)])
        dh = np.sum(inner_prods[0])
        hh = np.real(np.sum(inner_prods[1]))

        if self.marg_phi_ref:
            dh  = np.abs(dh)
        else:
            dh  = np.real(dh)

        return dh, hh, self.dd

    def log_like(self, params):
        """
            log-likelihood function,
            params : current sample as dictionary (filled from Prior)
        """

        #generate waveform
        hphc    = np.array(self.wave.compute_hphc(params))

        # dh , hh
        inner_prods = np.transpose([list(self.inner_products_singleifo(i, ifo, params, hphc)) for i,ifo in enumerate(self.ifos)])
        dh = np.sum(inner_prods[0])
        hh = np.real(np.sum(inner_prods[1]))

        if self.marg_phi_ref:
            dh  = np.abs(dh)
            R   = dh + np.log(i0e(dh))
        else:
            dh  = np.real(dh)
            R   = dh

        lnl = R - 0.5*hh

        return np.real(lnl)

    # compute (d|h) and (h|h)
    def prods_sdat(self, k, rdata):
        """
            Compute products (h|h), (d|h) for the k-th detector using summary data,
            for logL evalutation with marginalized phi_ref
        """
        r0, r1 = rdata

        # compute logL components
        hh      = self.sdat[2][k]*np.absolute(r0)**2 + self.sdat[3][k]*2.0*(r0*np.conjugate(r1)).real
        dh      = self.sdat[0][k]*np.conjugate(r0) + self.sdat[1][k]*np.conjugate(r1)
        return trapz(dh, x=self.fax), trapz(np.real(hh), x=self.fax)

    # compute relative waveform r(f) = h(f)/h0(f)
    def compute_rf(self, h, i):

        """
            compute the ratio r(f) = h(f)/h0(f) where h0(f) is some fiducial waveform and h(f) correspond to parameter combinations par
            h : current waveform, already sampled at fbin and projected on the detector
            h0: fiducial waveform (it is important to pass one that is NOT shifted to the right merger time)
            fbin: frequency bin edges
            par is list of parameters: [Mc, eta, chieff, chia, Lam, dtc]
            tc: best-fit time
            """

        f       = self.fbin
        h0_bin  = self.h0_bin[i]

        # waveform ratio
        r   = h / h0_bin
        r0  = 0.5*(r[:-1] + r[1:])
        r1  = (r[1:] - r[:-1])/(f[1:] - f[:-1])

        return np.array([r0, r1], dtype=np.complex128)

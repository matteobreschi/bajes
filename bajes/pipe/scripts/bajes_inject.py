#!/usr/bin/env python
from __future__ import division
import os
import numpy as np
import optparse as op

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass

from bajes.obs.gw.noise import Noise
from bajes.obs.gw.waveform import Waveform
from bajes.obs.gw.strain import Series, fft, ifft
from bajes.obs.gw.detector import Detector, calc_project_array_td
from bajes.obs.gw.utils import read_asd, read_params
from bajes.pipe import set_logger, ensure_dir

def make_spectrogram_plot(ifo, time, inj_strain, noise, f_min, outdir):

    dt      = np.median(np.diff(time))
    srate   = 1./dt
    fNyq    = srate/2.
    freq , inj_hfft = fft(inj_strain,dt)
    seglen  = 1./np.median(np.diff(freq))
    asd     = noise.interp_asd_pad(freq)
    inj_hfft_whit = inj_hfft/(asd*np.sqrt(fNyq))

    time_whit, inj_strain_whit = ifft(inj_hfft_whit , srate=srate, seglen=seglen)

    Nfft    = int (fNyq)//2
    Novl    = int (Nfft * 0.99)
    window  = np.blackman(Nfft)

    try:
        fig = plt.figure(figsize=(12,9))
        plt.title("{} spectrogram".format(ifo), size = 14)
        spec, freqs, bins, im = plt.specgram(inj_strain_whit, NFFT=int(Nfft), Fs=int(opts.srate), noverlap=int(Novl),
                                             window=window, cmap='PuBu', xextent=[0,seglen])
        plt.yscale('log')
        plt.ylim((f_min,0.5/dt))
        plt.xlabel("time [s]")
        plt.ylabel("frequency [Hz]")
        plt.savefig(outdir + '/{}_spectrogram.png'.format(ifo), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


def make_injection_plot(ifo, time, inj_strain, wave_strain, noise, f_min, outdir):

    try:
        fig = plt.figure(figsize=(12,9))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # plot injected strain (black) and signal (red)
        ax1.set_title("{} injection".format(ifo), size = 14)
        ax1.plot(time , inj_strain, c='gray', lw=0.7, label='Injected strain')
        ax1.plot(time , wave_strain, c='slateblue', label='Projected wave')
        ax1.legend(loc='best')
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('strain')

        # plot central 1s of signal
        mask_ax2 = np.where((time>=np.median(time)-0.5)&(time<=np.median(time)+0.5))
        ax2.plot(time[mask_ax2], wave_strain[mask_ax2], c='r')

        plt.savefig(outdir + '/{}_strains.png'.format(ifo), dpi=100, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


    from scipy.signal import tukey

    dt                   = np.median(np.diff(time))
    seglen               = time[-1]-time[0]
    freq_proj, hfft_proj = fft(wave_strain, dt)
    freq_inj, hfft_inj   = fft(inj_strain*tukey(len(inj_strain),alpha=0.4/seglen), dt)

    try:
        fig = plt.figure(figsize=(12,9))
        ax1 = fig.add_subplot(111)

        # plot injected strain (black) and signal (red)
        ax1.set_title("{} spectra".format(ifo), size = 14)
        ax1.loglog(freq_inj , np.abs(hfft_inj), c='gray', lw=0.7, label='Injected strain')
        ax1.loglog(freq_proj , np.abs(hfft_proj), c='royalblue', label='Projected wave')
        ax1.loglog(freq_inj , noise.interp_asd_pad(freq_inj), c='navy', label='ASD')
        ax1.legend(loc='best')

        ax1.set_xlim((f_min,1./dt/2.))
        ax1.set_xlabel('frequency [Hz]')
        ax1.set_ylabel('amplitude spectrum')

        plt.savefig(outdir + '/{}_spectra.png'.format(ifo), dpi=100, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

class Injection(object):

    def __init__(self, ifos, dets, noises, data_path, seglen, srate, f_min, ra, dec, psi, t_gps, wind_flag, zero_flag,
                 tukey=0.1, seed=None):

        self.ifos   = ifos
        self.dets   = dets
        self.noises = noises
        if len(self.ifos) != len(self.dets.keys()):
            raise ValueError("")
        elif len(self.ifos) != len(self.noises.keys()):
            raise ValueError("")

        self.seglen = seglen
        self.srate  = srate
        self.f_min  = f_min
        self.ra     = ra
        self.dec    = dec
        self.psi    = psi
        self.t_gps  = t_gps
        self.tukey  = tukey
        self.snrs   = {}

        # Generate noise-only
        if not data_path:
            logger.info("No data file was passed, generating pure noise.")

            self.wave_strains   = {}
            self.noise_strains  = {}
            self.inj_strains    = {}
            self.times          = {}
            Npt                 = int(self.seglen*self.srate)

            for ifo in self.ifos:

                self.wave_strains[ifo]  = np.zeros(Npt)
                if not zero_flag:
                    self.noise_strains[ifo] = noises[ifo].generate_fake_noise(self.seglen, self.srate, self.t_gps, filter=True)
                    self.inj_strains[ifo]   = self.noise_strains[ifo] + self.wave_strains[ifo]
                else:
                    raise Exception("A zero noise injection was selected and no data file was provided, thus the strain is made of zeros only.")
                self.times[ifo]             = np.arange(Npt,dtype=float)/srate - self.seglen/2 + self.t_gps

                # compute SNR
                self.snrs[ifo] = 0.
                logger.info("  - SNR in {} = {:.3f} ".format(ifo, self.snrs[ifo]))

        else:

            self.data_path  = os.path.abspath(data_path)
            tag             = self.data_path.split('.')[-1]

            if seed is not None:
                np.random.seed(seed)

            # read injection section using txt/dat polarizations
            if tag == 'txt' or tag == 'dat':

                logger.info("... reading polarizations from ascii file ...")
                times, hp, hc = np.genfromtxt(self.data_path, usecols=[0,1,2], unpack=True)

                from scipy.signal import tukey
                if wind_flag == 'low':
                    window  = tukey(len(hp), alpha=0.4/seglen)
                    imin    = np.max(np.where(window==1))
                    for i in range(len(window)):
                        if i >= imin:
                            window[i] = 1.
                elif wind_flag == 'both':
                    window  = tukey(len(hp), alpha=0.4/seglen)
                elif wind_flag == 'none':
                    window  = np.ones(len(hp))
                else:
                    logger.error("Invalid window flag passed to the injection. Please use 'low', 'both' or 'none'.")
                    raise ValueError("Invalid window flag passed to the injection. Please use 'low', 'both' or 'none'.")

                hp = hp * window
                hc = hc * window

                # create series in order to cut the strain
                # obs. avoid window since it is already applied
                ser_hp  = Series('time', hp, srate=self.srate, seglen=self.seglen,
                                 f_min=self.f_min, f_max=self.srate/2,
                                 t_gps=self.t_gps, only=True, alpha_taper=0.0)
                ser_hc  = Series('time', hc, srate=self.srate, seglen=self.seglen,
                                 f_min=self.f_min, f_max=self.srate/2,
                                 t_gps=self.t_gps, only=True, alpha_taper=0.0)

                hp = ser_hp.time_series
                hc = ser_hc.time_series
                assert len(hp)==len(hc)

                # estimate actual GPS time of merger
                amp     = np.abs(hp - 1j*hc)
                dt_mrg  = np.argmax(amp)*ser_hp.dt
                t_gps_mrg = self.t_gps - self.seglen/2. + dt_mrg

                self.wave_strains   = {}
                self.noise_strains  = {}
                self.inj_strains    = {}
                self.times          = {}

                for ifo in self.ifos:

                    self.wave_strains[ifo]  = calc_project_array_td(self.dets[ifo], hp, hc,
                                                                    1/self.srate,self.ra,self.dec,self.psi,t_gps_mrg)
                    if not zero_flag:
                        self.noise_strains[ifo] = noises[ifo].generate_fake_noise(self.seglen, self.srate, self.t_gps, filter=True)
                        self.inj_strains[ifo]   = self.noise_strains[ifo] + self.wave_strains[ifo]
                    else:
                        self.inj_strains[ifo]   = self.wave_strains[ifo]
                    self.times[ifo]             = ser_hp.times

                    # compute SNR
                    _f, _w = fft(self.wave_strains[ifo], 1./self.srate)
                    _f, _d = fft(self.inj_strains[ifo], 1./self.srate)
                    _i     = np.where(_f>=self.f_min)
                    psd    = noises[ifo].interp_psd_pad(_f[_i])
                    d_inner_h = (4/self.seglen)*np.real(np.sum(np.conj(_w[_i])*_d[_i]/psd))
                    h_inner_h = (4/self.seglen)*np.real(np.sum(np.conj(_w[_i])*_w[_i]/psd))
                    self.snrs[ifo] = d_inner_h/np.sqrt(h_inner_h)
                    logger.info("  - SNR in {} = {:.3f} ".format(ifo, self.snrs[ifo]))

            # read injection section of params.ini and generate TDWaveform
            elif tag == 'ini':

                logger.info("... generating polarizations from config file ...")

                params = read_params(self.data_path, 'injection')

                if 'approx' not in params.keys():
                    raise AttributeError("Impossible to generate the waveform for injection. Approximant field is missing.")

                # check skylocation,
                # overwrite command-line input if skyloc is in params.ini
                if 'ra' in list(params.keys()):
                    logger.info("Overriding the right ascension value found in the parameter file.")
                    params['ra']    = self.ra
                if 'dec' in list(params.keys()):
                    logger.info("Overriding the declination value found in the parameter file.")
                    params['dec']   = self.dec
                if 'psi' in list(params.keys()):
                    logger.info("Overriding the polarisation value found in the parameter file.")
                    params['psi']   = self.psi

                list_params, list_values = [], []
                for key in params.keys():
                    list_params.append(key)
                    list_values.append(params[key])
                logger.info("Generating injection with the paramters:")
                for i in range(len(list_params)):
                    logger.info("{}: {}".format(list_params[i], list_values[i]))

                # fix missing information
                params['f_min']     = self.f_min
                params['f_max']     = self.srate/2
                params['seglen']    = self.seglen
                params['srate']     = self.srate
                params['t_gps']     = self.t_gps
                params['tukey']     = self.tukey

                params_keys = list(params.keys())

                if 'approx' not in params_keys:
                    logger.error("Unspecified approximant model for gravitational-wave injection")
                    raise RuntimeError("Unspecified approximant model for gravitational-wave injection")

                if ('mchirp' not in params_keys) and ('mtot' not in params_keys):
                    logger.error("Unspecified total mass / chirp mass parameter for gravitational-wave injection")
                    raise RuntimeError("Unspecified total mass / chirp mass parameter for gravitational-wave injection")

                if 'q' not in params_keys:
                    logger.error("Unspecified mass ratio parameter for gravitational-wave injection")
                    raise RuntimeError("Unspecified mass ratio parameter for gravitational-wave injection")

                wave        = Waveform(np.linspace(0,self.srate/2,int(self.seglen*self.srate)//2 +1),  self.srate, self.seglen, params['approx'])
                self.time   = wave.times - self.seglen/2 + self.t_gps

                if wave.domain == 'freq':
                    raise AttributeError("Selected waveform model ({}) exists only in frequency-domain. Please use time-domain approximant to perform the injection.".format(params['approx']))

                signal_template = wave.compute_hphc(params)
                hp      = signal_template.plus
                hc      = signal_template.cross
                series  = Series(wave.domain, hp, srate=self.srate,
                                 seglen=self.seglen, f_min=self.f_min,
                                 f_max=self.srate/2, t_gps=self.t_gps,
                                 alpha_taper=0.0)

                self.wave_strains   = {}
                self.noise_strains  = {}
                self.inj_strains    = {}
                self.times          = {}

                for ifo in self.ifos:

                    # initialize detector with empty data
                    self.dets[ifo].store_measurement(series, noises[ifo])
                    # compute the data
                    self.wave_strains[ifo]  = self.dets[ifo].project_tdwave(signal_template, params, wave.domain)
                    if not zero_flag:
                        self.noise_strains[ifo] = noises[ifo].generate_fake_noise(self.seglen, self.srate, self.t_gps, filter=True)
                        self.inj_strains[ifo]   = self.noise_strains[ifo] + self.wave_strains[ifo]
                    else:
                        self.inj_strains[ifo]   = self.wave_strains[ifo]
                    self.times[ifo]             = self.time

                    # re-initialize detector with actual data
                    self.dets[ifo].store_measurement(Series('time', self.inj_strains[ifo],
                                                            srate=self.srate, seglen=self.seglen, f_min=self.f_min,
                                                            f_max=self.srate/2, t_gps=self.t_gps, alpha_taper=0.0),
                                                     noises[ifo])

                    # compute SNR
                    d_inner_h, h_inner_h, d_inner_d = self.dets[ifo].compute_inner_products(signal_template, params, wave.domain)
                    d_inner_h = np.sum(d_inner_h)
                    self.snrs[ifo] = d_inner_h/np.sqrt(h_inner_h)
                    logger.info("  - SNR in {} = {:.3f} ".format(ifo, self.snrs[ifo]))

            else:

                logger.error("Impossible to generate injection from {} file. Use txt/dat or ini.".format(tag))
                ValueError("Impossible to generate injection from {} file. Use txt/dat or ini.".format(tag))

        # print network SNR
        if list(self.snrs.keys()):
            net_snr = np.sqrt(sum([ self.snrs[ifo]**2. for ifo in  list(self.snrs.keys()) ]))
            logger.info("  - SNR in the Network = {:.3f} ".format(net_snr))

    def write_injections(self, outdir):
        for ifo in self.ifos:

            # write signal
            if len(list(self.wave_strains.keys()))>0:
                injectionfile = open(outdir + '/{}_signal.txt'.format(ifo), 'w')
                injectionfile.write('#time\t strain\n')
                for tii,hii in zip(np.array(self.times[ifo],dtype=float), np.array(self.wave_strains[ifo],dtype=float)):
                    injectionfile.write('{}\t{}\n'.format(tii, hii))
                injectionfile.close()

            # write noie
            if len(list(self.noise_strains.keys()))>0:
                injectionfile = open(outdir + '/{}_noise.txt'.format(ifo), 'w')
                injectionfile.write('#time\t strain\n')
                for tii,hii in zip(np.array(self.times[ifo],dtype=float),  np.array(self.noise_strains[ifo],dtype=float)):
                    injectionfile.write('{}\t{}\n'.format(tii, hii))
                injectionfile.close()

            # write data
            injectionfile = open(outdir + '/{}_INJECTION.txt'.format(ifo), 'w')
            injectionfile.write('#time\t strain\n')
            for tii,hii in zip(np.array(self.times[ifo],dtype=float), np.array(self.inj_strains[ifo],dtype=float)):
                injectionfile.write('{}\t{}\n'.format(tii, hii))
            injectionfile.close()

            # plots
            make_injection_plot(ifo, self.times[ifo] , self.inj_strains[ifo], self.wave_strains[ifo], self.noises[ifo], self.f_min, outdir )
            make_spectrogram_plot(ifo, self.times[ifo], self.inj_strains[ifo], self.noises[ifo], self.f_min, outdir)

def bajes_inject_parser():

    parser=op.OptionParser()
    parser.add_option('--ifo',         dest='ifos',                       type='string',  action="append",    help="Single IFO tag. This option needs to be passed separately for every ifo in which the injection is requested. The order must correspond to the one in which the '--asd' commands are passed. Available options: ['H1', 'L1', 'V1', 'K1', 'G1'].")
    parser.add_option('--asd',         dest='asds',                       type='string',  action="append",    help="Single path to ASD file. This option needs to be passed separately for every ifo in which the injection is requested.  The order must correspond to the one in which the '--ifo' commands are passed.")

    parser.add_option('--wave',     dest='wave',        type='string',  default='',     help='path to strain data to inject, the file should contains 3 cols [t, reh, imh]. If empty, pure noise is generated.')
    parser.add_option('--srate',    dest='srate',       type='float',   help='sampling rate of the injected waveform [Hz] and it will be the srate of the sampling, please check that everything is consistent')
    parser.add_option('--seglen',   dest='seglen',      type='float',   help='length of the segment of the injected waveform [sec], if it is not a power of 2, the final segment will be padded')

    parser.add_option('--f-min',    dest='f_min',       type='float',   default=20.,            help='minimum frequency [Hz], default 20Hz')
    parser.add_option('--t-gps',    dest='t_gps',       type='float',   default=1187008882,     help='GPS time of the series, default 1187008882 (GW170817)')

    parser.add_option('--zero-noise',    dest='zero',   action="store_true",    default=False,  help='use zero noise')
    parser.add_option('--seed',          dest='seed',   type='int',             default=None,   help='seed for random number generator')

    parser.add_option('--ra',          dest='ra',     default=None,       type='float',                       help='right ascencion location of the injected source. Default optimal location for the first IFO.')
    parser.add_option('--dec',         dest='dec',    default=None,       type='float',                       help='declination location of the injected source. Default optimal location for the first IFO.')
    parser.add_option('--pol',         dest='psi',    default=0.,         type='float',                       help='polarization angle of the injected source. Default: 0.')
    parser.add_option('--window',      dest='window', default='low',      type='string',                      help="Location of the window. Available options: ['low' or 'both']. Default: 'low'.")

    parser.add_option('--tukey',       dest='tukey',  default=0.1,        type='float',                       help='tukey window parameter')

    parser.add_option('-o','--outdir', dest='outdir', default=None,       type='string',                      help='Output directory. Default: None.')
    return parser.parse_args()

if __name__ == "__main__":

    (opts,args) = bajes_inject_parser()
    dets    = {}
    noises  = {}

    if opts.outdir == None:
        raise ValueError("Passing an output directory is mandatory. Please pass a value through the '--outdir' option.")

    opts.outdir = os.path.abspath(opts.outdir)
    ensure_dir(opts.outdir)

    global logger
    logger = set_logger(outdir=opts.outdir, label='bajes_inject')
    logger.info("Running bajes inject:")

    for i in range(len(opts.ifos)):
        ifo = opts.ifos[i]
        logger.info("... setting detector {} for injection ...".format(ifo))
        if not(ifo in opts.asds[i]):
            logger.info("WARNING: the name of ASD file does not correspond to selected IFO. Please check that IFO and ASD were passed in the right order.")
        dets[ifo]   = Detector(ifo,opts.t_gps)
        fr,asd      = read_asd(opts.asds[i], ifo)
        noises[ifo] = Noise(fr, asd, f_min = opts.f_min, f_max = opts.srate/2.)

    if opts.ra == None and opts.dec == None:
        logger.info("... no input sky position, using optimal location for {} ...".format(opts.ifos[0]))
        opts.ra , opts.dec = dets[opts.ifos[0]].optimal_orientation(opts.t_gps)
    else:
        logger.info("... locating the source at input sky position ...")

    logger.info("  - right ascenscion = {:.3f}".format(opts.ra))
    logger.info("  - declination = {:.3f}".format(opts.dec))

    logger.info("... injecting waveform into detectors ...")
    if not opts.wave:
        logger.warning("Waveform option is empty, generating pure noise injection.")
    inj = Injection(opts.ifos, dets, noises, opts.wave, opts.seglen, opts.srate, opts.f_min,
                    opts.ra, opts.dec, opts.psi, opts.t_gps, opts.window, opts.zero, opts.tukey, opts.seed)

    logger.info("... writing strain data files ...")
    inj.write_injections(opts.outdir)


    logger.info("... waveform injected.")

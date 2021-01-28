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

def make_spectrogram_plot(ifo, time, inj_strain, noise, outdir):

    dt      = np.median(np.diff(time))
    srate   = 1./dt
    fNyq    = srate/2.
    freq , inj_hfft = fft(inj_strain,dt)
    seglen  = 1./np.median(np.diff(freq))
    asd     = noise.interp_asd_pad(freq)
    inj_hfft_whit = inj_hfft/(asd*np.sqrt(fNyq))

    time_whit, inj_strain_whit = ifft(inj_hfft_whit , srate)

    Nfft    = int (fNyq)//2
    Novl    = int (Nfft * 0.9)
    window  = np.blackman(Nfft)

    fig = plt.figure(figsize=(12,9))
    plt.title("{} spectrogram".format(ifo), size = 14)
    spec, freqs, bins, im = plt.specgram(inj_strain_whit, NFFT=int(Nfft), Fs=int(opts.srate), noverlap=int(Novl),
                                         window=window, cmap='viridis', xextent=[0,seglen])
    plt.yscale('log')
    plt.ylim((20,2000))
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    plt.savefig(outdir + '/{}_spectrogram.png'.format(ifo), dpi=200)
    plt.close()

def make_injection_plot(ifo, time, inj_strain, wave_strain, noise, outdir):

    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # plot injected strain (black) and signal (red)
    ax1.set_title("{} injection".format(ifo), size = 14)
    ax1.plot(time , inj_strain, c='k', lw=0.7, label='injected strain')
    ax1.plot(time , wave_strain, c='r', label='projected wave')
    ax1.legend(loc='best')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('strain')
    
    # plot central 1s of signal
    mask_ax2 = np.where((time>=np.median(time)-0.5)&(time<=np.median(time)+0.5))
    ax2.plot(time[mask_ax2], wave_strain[mask_ax2], c='r')
    
    plt.savefig(outdir + '/{}_strains.png'.format(ifo), dpi=250)
    plt.close()
    
    from scipy.signal import tukey
    
    dt = np.median(np.diff(time))
    freq_proj, hfft_proj = fft(wave_strain, dt)
    freq_inj, hfft_inj = fft(inj_strain*tukey(len(inj_strain),alpha=0.1), dt)

    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)
    
    # plot injected strain (black) and signal (red)
    ax1.set_title("{} spectra".format(ifo), size = 14)
    ax1.loglog(freq_inj , np.abs(hfft_inj), c='k', lw=0.7, label='injected strain')
    ax1.loglog(freq_proj , np.abs(hfft_proj), c='r', label='projected wave')
    ax1.loglog(freq_inj , noise.interp_asd_pad(freq_inj), c='g', label='ASD')
    ax1.legend(loc='best')
    
    ax1.set_xlim((10,1./dt/2.))
    ax1.set_xlabel('frequency [Hz]')
    ax1.set_ylabel('amplitude spectrum')
    
    plt.savefig(outdir + '/{}_spectra.png'.format(ifo), dpi=250)
    plt.close()

class Injection(object):

    def __init__(self, ifos, dets, noises, data_path, seglen, srate, f_min, ra, dec, psi, t_gps, wind_flag, zero_flag, tukey=0.1):

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
        
        self.data_path  = os.path.abspath(data_path)
        tag             = self.data_path.split('.')[-1]
        
        # read injection section using txt/dat polarizations
        if tag == 'txt' or tag == 'dat':
            
            logger.info("... reading polarizations from ascii file ...")
            times, hp, hc = np.genfromtxt(self.data_path, usecols=[0,1,2], unpack=True)
        
            from scipy.signal import tukey
            if wind_flag == 'low':
                window  = tukey(len(hp), alpha=0.1)
                imin    = np.max(np.where(window==1))
                for i in range(len(window)):
                    if i >= imin:
                        window[i] = 1.
            elif wind_flag == 'both':
                window  = tukey(len(hp), alpha=0.1)
            elif wind_flag == 'none':
                window  = np.ones(len(hp))
            else:
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
                    self.noise_strains[ifo] = noises[ifo].generate_fake_noise(self.seglen, self.srate, self.t_gps)
                    self.inj_strains[ifo]   = self.noise_strains[ifo] + self.wave_strains[ifo]
                else:
                    self.inj_strains[ifo]   = self.wave_strains[ifo]
                
                self.times[ifo]             = ser_hp.times
            
        # read injection section of params.ini and generate TDWaveform
        elif tag == 'ini':
            
            logger.info("... generating polarizations from config file ...")

            params = read_params(self.data_path, 'injection')
            
            if 'approx' not in params.keys():
                raise AttributeError("Impossible to generate the waveform for injection. Approximant field is missing.")
            
            # check skylocation
            if 'ra' not in list(params.keys()):
                logger.info("... right ascension found in parameter file ...")
                params['ra']    = self.ra
            if 'dec' not in list(params.keys()):
                logger.info("... declination found in parameter file ...")
                params['dec']   = self.dec
            if 'psi' not in list(params.keys()):
                logger.info("... polarization found in parameter file ...")
                params['psi']   = self.psi
            
            # fix missing information
            params['f_min']     = self.f_min
            params['f_max']     = self.srate/2
            params['seglen']    = self.seglen
            params['srate']     = self.srate
            params['t_gps']     = self.t_gps
            params['tukey']     = self.tukey
            
            params_keys = list(params.keys())
            
            if 'approx' not in params_keys:
                raise RuntimeError("Unspecified approximant model for gravitational-wave injection")

            if ('mchirp' not in params_keys) and ('mtot' not in params_keys):
                raise RuntimeError("Unspecified total mass / chirp mass parameter for gravitational-wave injection")

            if 'q' not in params_keys:
                raise RuntimeError("Unspecified mass ratio parameter for gravitational-wave injection")

            wave        = Waveform(np.linspace(0,self.srate/2,int(self.seglen*self.srate)//2 +1),  self.srate, self.seglen, params['approx'])
            self.time   = wave.times - self.seglen/2 + self.t_gps
            
            if wave.domain == 'freq':
                raise AttributeError("Selected waveform model exists only in frequency-domain. Please use time-domain approximant to perform the injection.")
            
            signal_template  = wave.compute_hphc(params)
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
                
                self.dets[ifo].store_measurement(series, noises[ifo])
                self.wave_strains[ifo]  = self.dets[ifo].project_tdwave(signal_template, params, wave.domain)
                if not zero_flag:
                    self.noise_strains[ifo] = noises[ifo].generate_fake_noise(self.seglen, self.srate, self.t_gps)
                    self.inj_strains[ifo]   = self.noise_strains[ifo] + self.wave_strains[ifo]
                else:
                    self.inj_strains[ifo]   = self.wave_strains[ifo]
                self.times[ifo]             = self.time

        else:
            
            logger.error("Impossible to generate injection from {} file. Use txt/dat or ini.".format(tag))
            ValueError("Impossible to generate injection from {} file. Use txt/dat or ini.".format(tag))
        
    def write_injections(self, outdir):
        for ifo in self.ifos:
            injectionfile = open(outdir + '/{}_INJECTION.txt'.format(ifo), 'w')
            injectionfile.write('# time \t strain \n')
            for i in range(len(self.inj_strains[ifo])):
                injectionfile.write('{:.15f} \t {} \n'.format(self.times[ifo][i], self.inj_strains[ifo][i]))
            injectionfile.close()

            make_injection_plot(ifo, self.times[ifo] , self.inj_strains[ifo], self.wave_strains[ifo], self.noises[ifo], outdir )
            make_spectrogram_plot(ifo, self.times[ifo], self.inj_strains[ifo], self.noises[ifo], outdir)


if __name__ == "__main__":
    
    parser=op.OptionParser()
    parser.add_option('--ifo',      dest='ifos',        type='string',  action="append", help='IFO tag, i.e. H1, L1, V1, K1, G1')
    parser.add_option('--asd',      dest='asds',        type='string',  action="append", help='path to ASD files')

    parser.add_option('--wave',     dest='wave',        type='string',   help='path to strain data to inject, the file should contains 3 cols [t, reh, imh]')
    parser.add_option('--srate',    dest='srate',       type='float',   help='sampling rate of the injected waveform [Hz] and it will be the srate of the sampling, please check that everything is consistent')
    parser.add_option('--seglen',   dest='seglen',      type='float',   help='length of the segment of the injected waveform [sec], if it is not a power of 2, the final segment will be padded')
    
    parser.add_option('--f-min',    dest='f_min',       type='float',   default=20.,            help='minimum frequency [Hz], default 20Hz')
    parser.add_option('--t-gps',    dest='t_gps',       type='float',   default=1187008882,    help='GPS time of the series, default 1187008882 (GW170817)')
    
    parser.add_option('--zero-noise',    dest='zero',   action="store_true",   default=False,            help='use zero noise')

    parser.add_option('--ra',   dest='ra',      default=None,   type='float',   help='right ascencion location of the injected source, default best location for first IFO.')
    parser.add_option('--dec',  dest='dec',     default=None,   type='float',   help='declination location of the injected source, default best location for first IFO.')
    parser.add_option('--pol',  dest='psi',     default=0.,     type='float',   help='polarization angle of the injected source, default 0.')
    parser.add_option('--window',  dest='window',     default='low',     type='string',   help='location of the window, low or both')
    
    parser.add_option('--tukey',  dest='tukey',     default=0.1,     type='float',   help='tukey window parameter')
    
    parser.add_option('-o','--outdir',default=None,type='string',dest='outdir',help='output directory')
    (opts,args) = parser.parse_args()

    dets    = {}
    noises  = {}
    
    if opts.outdir == None:
        raise ValueError("Unable to ensure output directory.")
    
    opts.outdir = os.path.abspath(opts.outdir)
    ensure_dir(opts.outdir)
    
    global logger
    logger = set_logger(outdir=opts.outdir, label='bajes_inject')
    logger.info("Running bajes inject:")
    
    for i in range(len(opts.ifos)):
        ifo = opts.ifos[i]
        logger.info("... setting detector {} for injection ...".format(ifo))
        dets[ifo]   = Detector(ifo,opts.t_gps)
        fr,asd      = read_asd(opts.asds[i], ifo)
        noises[ifo] = Noise(fr,asd)

    if opts.ra == None and opts.dec == None:
        logger.info("... no input sky position, using optimal location for {} ...".format(opts.ifos[0]))
        opts.ra , opts.dec = dets[opts.ifos[0]].optimal_orientation(opts.t_gps)
    else:
        logger.info("... locating the source at input sky position ...")

    logger.info("  - right ascenscion = {:.3f}".format(opts.ra))
    logger.info("  - declination = {:.3f}".format(opts.dec))

    logger.info("... injecting waveform into detectors ...")
    inj = Injection(opts.ifos, dets, noises, opts.wave, opts.seglen, opts.srate, opts.f_min,
                    opts.ra, opts.dec, opts.psi, opts.t_gps, opts.window, opts.zero, opts.tukey)

    logger.info("... writing strain data files ...")
    inj.write_injections(opts.outdir)


    logger.info("... waveform injected.")

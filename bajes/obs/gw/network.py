import numpy as np
from . import Detector, Noise, Series, Waveform
from .utils import read_asd

import logging
logger = logging.getLogger(__name__)

class Network:
    
    ###
    #-----init-----
    def __init__(self, ifos, t_gps):
        self.ifos        = ifos
        self.t_gps       = t_gps
        self.detectors   = [ Detector(ifo, t_gps=t_gps) for ifo in ifos ]
        self.noise_ifos  = None
        self.series_ifos = None
        self.wave        = None

    ###
    #-----noise-----
    def prep_noise(self, noise_dict):
        self.noise_ifos = { ifo : Noise(*read_asd(noise_dict['event'], ifo), **noise_dict['settings']) for ifo in self.ifos }

    ###
    #-----series-----
    def prep_series(self, series_dict):
        assert series_dict['settings']['t_gps'] == self.t_gps, f"t_gps in Network ({self.t_gps}) is different from the series input ({series_dict['settings']['t_gps']})!"

        self.series_ifos = { ifo : None for ifo in self.ifos }
        for ifo in self.ifos:
            times, strain = np.genfromtxt(series_dict[ifo]['data_path'], usecols=series_dict[ifo]['usecols'], unpack=series_dict[ifo]['unpack'])
            self.series_ifos[ifo] = Series(series_dict['domain'], strain, **series_dict['settings'])
            assert 1/(times[1]-times[0]) == self.series_ifos[ifo].srate, "Series settings do not match with time information from data file!"
            if series_dict[ifo]['bandpassing'] is not None: self.series_ifos[ifo].bandpassing(**series_dict[ifo]['bandpassing'])

    ###
    #-----noise + series-----
    def whitening(self):
        assert self.series_ifos is not None and self.noise_ifos is not None, "Prepare noise and series before whitening!"
        for ifo in self.ifos:
            self.series_ifos[ifo].whitening(self.noise_ifos[ifo])

    def store_measurement(self):
        assert (self.series_ifos is not None and self.noise_ifos is not None), "Prepare noise and series before storing measurement!"
        for det in self.detectors:
            det.store_measurement(self.series_ifos[det.ifo], self.noise_ifos[det.ifo])        

    ###
    #-----wave-----
    def prep_wave(self, approx):
        assert self.series_ifos is not None, "Prepare series before wave!"
        assert np.all(np.array([ self.series_ifos[self.ifos[i]].freqs == self.series_ifos[self.ifos[j]].freqs for i,j in zip(range(0,len(self.ifos) - 1), range(1,len(self.ifos))) ])), "Mismatch in series.freqs for the various ifos!"
        self.wave = Waveform(self.series_ifos[self.ifos[0]].freqs, self.series_ifos[self.ifos[0]].srate, self.series_ifos[self.ifos[0]].seglen, approx)
        
    def eval_wave(self, params):
        assert self.wave is not None, "Prepare wave before evaluating!"
        return self.wave.compute_hphc(params)

    def proj_fdwave(self, params):
        assert self.wave is not None, "Prepare wave before projecting!"
        hpc = self.eval_wave(params)
        return { det.ifo : det.project_fdwave(hpc, params, self.wave.domain) for det in self.detectors }

    def proj_tdwave(self, params):
        assert self.wave is not None, "Prepare wave before projecting!"
        hpc = self.eval_wave(params)
        return { det.ifo : det.project_tdwave(hpc, params, self.wave.domain) for det in self.detectors }


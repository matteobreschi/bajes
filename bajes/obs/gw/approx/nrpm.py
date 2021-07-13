from __future__ import division, absolute_import
import numpy as np

from .nrpmw import NRPMw

MSUN_SI     = 1.9885469549614615e+30
MTSUN_SI    = 4.925491025543576e-06
MRSUN_SI    = 1476.6250614046494
PC_SI       = 3.085677581491367e+16

from ..utils import lambda_2_kappa
from ..utils.nrfits import bns_postmerger_amplitude, bns_postmerger_time, bns_postmerger_frequency

def NRPM_PhaseModel0( ti , fmerg , dfmerg , f0 , t0):
    return np.pi*ti*(6* np.power(t0,3)*(dfmerg*ti + 2*fmerg) + 4*t0* np.power(ti,2)*(-2*dfmerg*t0 + 3*f0 - 3*fmerg) + 3* np.power(ti,3)*(dfmerg*t0 - 2*f0 + 2*fmerg))/(6* np.power(t0,3))

def NRPM_PhaseModel1(  ti ,   t0 ,   f0):
    return 2* np.pi*f0*(ti-t0)

def NRPM_PhaseModel2( ti , f0 , f3 , t1 , t2):
    return np.pi*(-f0*t1 + f0*ti - f3*t1 + f3*ti + (f0 - f3)*(-t1 + t2)* np.sin( np.pi*(t1 - ti)/(t1 - t2))/ np.pi )

def NRPM_PhaseModel3( ti , f2 , f3, t2 , t3):
    return np.pi*(-f3*t2 + f3*ti - f2*t2 + f2*ti + (f3 - f2)*(-t2 + t3)* np.sin( np.pi*(t2 - ti)/(t2 - t3))/ np.pi )

def NRPM_PhaseModel4( ti , f2 , t3):
    return 2* np.pi*f2*(-t3 + ti)

def NRPM_AmpModel_plus( ti , b1 , b2 , t1, t2):

    bm  = (b1 + b2)/2
    bd  = (b2 - b1)/2
    w   = np.pi/(t2-t1)
    amp = bm + bd * np.cos(w*(ti-t1)+  np.pi)
    return amp

def NRPM_AmpModel_minus( ti , b1 , b2 , t1, t2):

    bm  = (b1 + b2)/2
    bd  = (b2 - b1)/2
    w   = np.pi/(t2-t1)
    amp = bm - bd * np.cos(w*(ti-t1))
    return amp

def NRPM_AmpModel_exp( ti , bmerg, b3 , t3 , tc, alpha = None):

    if alpha == None:
        alpha = np.log(100 * b3/bmerg)/(tc-t3)

    if alpha < 0:
        alpha = 0.

    amp = b3 * np.exp(- alpha * (ti - t3))
    return amp

def NRPM_AmpModel_exp2( ti , bmerg, b3 , t3 , tc, alpha = None):

    if alpha == None:
        a       = bmerg/(b3*100)
        alpha   = - (tc - t3)*np.log((1-a)/(1+a))

    amp = b3 * (2./(1.+np.exp(-alpha/(ti - t3))) - 1.)
    return amp

def NRPM_TaperBeforeMerger( ti , Mtot , eta , f_merg , A_merg ):


    # Use taper assuming chirp-like evolution
    Mchirp = Mtot * np.power(eta , 3./5.)

    t_merg  = (5./256.) * np.power(np.pi*f_merg,-8./3.) * np.power(Mchirp*MTSUN_SI,-5./3.)
    phase   = -2. * np.power(5. * Mchirp*MTSUN_SI,-5./8.) * np.power(t_merg - ti,5./8.)
    freq    = (1/np.pi) * np.power(5./(256. * (t_merg - ti)), 3./8.) * np.power(Mchirp * MTSUN_SI,-5./8.)
    dfreq   = (96./6.) * np.power(np.pi, 8./3.) * np.power(Mchirp * MTSUN_SI, 5./3.) * np.power( freq , 11./3.)

    # set taper 0.5ms before merger
    tcut = -0.0005
    indcut = np.where(ti>=tcut)

    ampl  = np.append(np.zeros(np.min(indcut)), A_merg * (1+np.cos(np.pi * ti[indcut]/tcut))/2)

    return ampl, phase, dfreq


def NRPM(srate, seglen, Mtot, q, kappa2T, distance, inclination, phi_merg,
         f_merg = None, alpha  = None, phi_kick = None):

    deltaT      = 1./srate
    time        = np.arange(int(seglen*srate))*deltaT - int(seglen*srate)*deltaT/2.

    eta         = q/((1+q)*(1+q))
    cosi        = np.cos(inclination)
    inc_plus    = (1.0+cosi*cosi)/2.0
    inc_cross   = cosi
    spherharm_norm = np.sqrt(5/(4*np.pi))
    distance    *= 1e6*PC_SI

    if kappa2T < 60 :
        return np.zeros(int(seglen*srate)), np.zeros(int(seglen*srate))

    # initialize hplus and hcross
    hplus = []
    hcross = []

    # Initialize amplitudes, times, freqs (label, kappa , mass, nu)
    a0 = bns_postmerger_amplitude(0, kappa2T, Mtot, eta)
    a1 = bns_postmerger_amplitude(1, kappa2T, Mtot, eta)
    a2 = bns_postmerger_amplitude(2, kappa2T, Mtot, eta)
    a3 = bns_postmerger_amplitude(3, kappa2T, Mtot, eta)
    am = bns_postmerger_amplitude('m', kappa2T, Mtot, eta)

    t0 = bns_postmerger_time(0, kappa2T, Mtot, eta)
    t1 = bns_postmerger_time(1, kappa2T, Mtot, eta)
    t2 = bns_postmerger_time(2, kappa2T, Mtot, eta)
    t3 = bns_postmerger_time(3, kappa2T, Mtot, eta)
    tc = bns_postmerger_time('e', kappa2T, Mtot, eta)

    f1 = bns_postmerger_frequency(1, kappa2T, Mtot, eta)
    f2 = bns_postmerger_frequency(2, kappa2T, Mtot, eta)
    f3 = bns_postmerger_frequency(3, kappa2T, Mtot, eta)

    if f_merg == None:
        fm = bns_postmerger_frequency('m', kappa2T, Mtot, eta)
    else:
        fm = f_merg

    # sanity check
    if f1 < fm:
        f1 = fm

    # listing indices
    indspre = np.where(time<=0)
    inds0   = np.where((time>0)&(time<t0))
    inds1   = np.where((time>=t0)&(time<t1))
    inds2   = np.where((time>=t1)&(time<t2))
    inds3   = np.where((time>=t2)&(time<t3))
    inds3a  = np.where((time>=t3)&(time<t3+3*MTSUN_SI*Mtot))
    inds4   = np.where(time>=t3+3*MTSUN_SI*Mtot)

    # compute taper pre-merger
    amppre, phipre, dfreq = NRPM_TaperBeforeMerger(time[indspre], Mtot , q, fm, am)
    phipre -= phipre[-1] + phi_merg

    hplus.append(spherharm_norm*inc_plus*amppre*np.cos(phipre)/distance)
    hcross.append(spherharm_norm*inc_cross*amppre*np.sin(phipre)/distance)

    amp0 = NRPM_AmpModel_plus(time[inds0], am , a0 , 0.0 , t0 )
    phi0 = NRPM_PhaseModel0(time[inds0],fm,dfreq[-1],f1,t0) + phi_merg
    hplus = np.append(hplus, spherharm_norm*inc_plus*amp0*np.cos(phi0)/distance)
    hcross = np.append(hcross, spherharm_norm*inc_cross*amp0*np.sin(phi0)/distance)

    if phi_kick == None:
        phi_kick = 0

    phi_0 = NRPM_PhaseModel0( t0, fm , dfreq[-1] , f1 , t0) + phi_merg + phi_kick

    amp1 = NRPM_AmpModel_minus(time[inds1], a0 , a1 , t0 , t1 )
    phi1 = NRPM_PhaseModel1(time[inds1], t0, f1) + phi_0
    hplus = np.append(hplus,spherharm_norm*inc_plus*amp1*np.cos(phi1)/distance)
    hcross = np.append(hcross,spherharm_norm*inc_cross*amp1*np.sin(phi1)/distance)

    phi_0 = NRPM_PhaseModel1(t1, t0, f1) + phi_0

    amp2 = NRPM_AmpModel_plus(time[inds2], a1 , a2 , t1 , t2 )
    phi2 = NRPM_PhaseModel2(time[inds2], f1 , f3 , t1, t2) + phi_0
    hplus = np.append(hplus,spherharm_norm*inc_plus*amp2*np.cos(phi2)/distance)
    hcross = np.append(hcross,spherharm_norm*inc_cross*amp2*np.sin(phi2)/distance)

    phi_0 = NRPM_PhaseModel2(t2, f1 , f3 , t1, t2) + phi_0

    amp3 = NRPM_AmpModel_minus(time[inds3], a2 , a3 , t2 , t3 )
    phi3 = NRPM_PhaseModel3(time[inds3], f2 , f3 , t2 , t3)  + phi_0
    hplus = np.append(hplus,spherharm_norm*inc_plus*amp3*np.cos(phi3)/distance)
    hcross = np.append(hcross,spherharm_norm*inc_cross*amp3*np.sin(phi3)/distance)

    phi_0 = NRPM_PhaseModel3(t3, f2 , f3 , t2 , t3) + phi_0

    amp3a = NRPM_AmpModel_minus(time[inds3a], a2 , a3 , t2 , t3 )
    phi3a = NRPM_PhaseModel4(time[inds3a], f2 , t3)  + phi_0
    hplus = np.append(hplus,spherharm_norm*inc_plus*amp3a*np.cos(phi3a)/distance)
    hcross = np.append(hcross,spherharm_norm*inc_cross*amp3a*np.sin(phi3a)/distance)

    amp4 = NRPM_AmpModel_exp(time[inds4], am , a3 , t3 , tc , alpha)
    phi4 = NRPM_PhaseModel4(time[inds4], f2 , t3) + phi_0
    hplus = np.append(hplus,spherharm_norm*inc_plus*amp4*np.cos(phi4)/distance)
    hcross = np.append(hcross,spherharm_norm*inc_cross*amp4*np.sin(phi4)/distance)

    return np.array(hplus), np.array(hcross)

def nrpm_wrapper(freqs, params):

    kappa2T = lambda_2_kappa(params['mtot']/(1.+1./params['q']),
                             params['mtot']/(1.+ params['q']),
                             params['lambda1'], params['lambda2'])

    hp, hc = NRPM(params['srate'], params['seglen'], params['mtot'], params['q'], kappa2T,
                  params['distance'], params['iota'], params['phi_ref'],
                  f_merg = None, alpha = None, phi_kick = None)

    return hp, hc

def nrpmw_wrapper(freqs, params):
    return NRPMw(freqs, params)

def nrpmw_recal_wrapper(freqs, params):
    return NRPMw(freqs, params, recalib=True)

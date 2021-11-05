from __future__ import division, absolute_import
import numpy as np

from .... import MSUN_SI, MTSUN_SI, MRSUN_SI
from .... import PARSEC_SI as PC_SI

__recalib_names__ = ['a0', 'a1', 'a2', 'a3', 'am', 't0', 't1', 't2', 't3', 'f1', 'f2', 'f3', 'fm']
__ERRS__ = {'a0':      0.5527843810575392,
            'a1':      0.08971658178398943,
            'a2':      0.23153047531626625,
            'a3':      0.18661598126308693,
            'am':      0.02461135167475065,
            't0':      0.0626817005527513,
            't1':      0.13951354610920094,
            't2':      0.33359134596173096,
            't3':      0.28587568610893244,
            'f1':      0.06513278643287955,
            'f2':      0.05115187749122023,
            'f3':      0.10417580565652033,
            'fm':      0.02868438522534468
}

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

def NRPM_PhaseModel4( ti , f2 , t3, beta = None):
    if beta == None:
        return 2*np.pi*f2*(-t3 + ti)
    else:
        tx = -t3 + ti
        return 2*np.pi*f2*tx*(1.+beta*tx)

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
         f_merg = None, alpha  = None, beta = None, phi_kick = None, recal = None):

    deltaT      = 1./srate
    Npt         = int(seglen*srate)
    time        = np.arange(Npt)*deltaT - Npt*deltaT/2.

    eta         = q/((1+q)*(1+q))
    cosi        = np.cos(inclination)
    inc_plus    = (1.0+cosi*cosi)/2.0
    inc_cross   = cosi
    spherharm_norm = np.sqrt(5/(4*np.pi))
    distance    *= 1e6*PC_SI

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

    # apply theretical recalibration
    if recal is not None:
            a0 *= (1. + recal['NRPM_recal_a0'])
            a1 *= (1. + recal['NRPM_recal_a1'])
            a2 *= (1. + recal['NRPM_recal_a2'])
            a3 *= (1. + recal['NRPM_recal_a3'])
            am *= (1. + recal['NRPM_recal_am'])

            t0 = max(0., t0*(1. + recal['NRPM_recal_t0']))
            t1 = max(t0, t1*(1. + recal['NRPM_recal_t1']))
            t2 = max(t1, t2*(1. + recal['NRPM_recal_t2']))
            t3 = max(t2, t3*(1. + recal['NRPM_recal_t3']))

            fm *= (1. + recal['NRPM_recal_fm'])
            f1 *= (1. + recal['NRPM_recal_f1'])
            f2 *= (1. + recal['NRPM_recal_f2'])
            f3 = max(f2, f3*(1. + recal['NRPM_recal_f3']))

    # listing indices
    indspre = np.where(time<=0)
    inds0   = np.where((time>0)&(time<t0))
    inds1   = np.where((time>=t0)&(time<t1))
    inds2   = np.where((time>=t1)&(time<t2))
    inds3   = np.where((time>=t2)&(time<t3))
    inds4   = np.where(time>=t3)

    # initialize hs
    hplus   = np.zeros(Npt)
    hcross  = np.zeros(Npt)

    # compute taper pre-merger
    amppre, phipre, dfreq = NRPM_TaperBeforeMerger(time[indspre], Mtot , q, fm, am)
    phipre -= phipre[-1] + phi_merg
    hplus[indspre]  = spherharm_norm*inc_plus*amppre*np.cos(phipre)/distance
    hcross[indspre] = spherharm_norm*inc_cross*amppre*np.sin(phipre)/distance

    phi_0 = phi_merg
    if t0 > 0:
        amp0 = NRPM_AmpModel_plus(time[inds0], am , a0 , 0.0 , t0 )
        phi0 = NRPM_PhaseModel0(time[inds0],fm,dfreq[-1],f1,t0) + phi_merg
        hplus[inds0]  = spherharm_norm*inc_plus*amp0*np.cos(phi0)/distance
        hcross[inds0] = spherharm_norm*inc_cross*amp0*np.sin(phi0)/distance
        phi_0 += NRPM_PhaseModel0( t0, fm , dfreq[-1] , f1 , t0)

    # if kappa2T < 60, it is assumed that
    # no post-merger radiation occurs due to prompt collapse
    if kappa2T < 60 :
        return hplus, hcross

    # additional phase-shift after merger
    if phi_kick == None:
        phi_kick = 0
    phi_0 += phi_kick

    if t1 > t0:
        amp1 = NRPM_AmpModel_minus(time[inds1], a0 , a1 , t0 , t1 )
        phi1 = NRPM_PhaseModel1(time[inds1], t0, f1) + phi_0
        hplus[inds1]  = spherharm_norm*inc_plus*amp1*np.cos(phi1)/distance
        hcross[inds1] = spherharm_norm*inc_cross*amp1*np.sin(phi1)/distance
        phi_0 += NRPM_PhaseModel1(t1, t0, f1)

    if t2 > t1:
        amp2 = NRPM_AmpModel_plus(time[inds2], a1 , a2 , t1 , t2 )
        phi2 = NRPM_PhaseModel2(time[inds2], f1 , f3 , t1, t2) + phi_0
        hplus[inds2]  = spherharm_norm*inc_plus*amp2*np.cos(phi2)/distance
        hcross[inds2] = spherharm_norm*inc_cross*amp2*np.sin(phi2)/distance
        phi_0 += NRPM_PhaseModel2(t2, f1 , f3 , t1, t2)

    if t3 > t2:
        amp3 = NRPM_AmpModel_minus(time[inds3], a2 , a3 , t2 , t3 )
        phi3 = NRPM_PhaseModel3(time[inds3], f2 , f3 , t2 , t3)  + phi_0
        hplus[inds3]  = spherharm_norm*inc_plus*amp3*np.cos(phi3)/distance
        hcross[inds3] = spherharm_norm*inc_cross*amp3*np.sin(phi3)/distance
        phi_0 += NRPM_PhaseModel3(t3, f2 , f3 , t2 , t3)

    amp4 = NRPM_AmpModel_exp(time[inds4], am , a3 , t3 , tc , alpha)
    phi4 = NRPM_PhaseModel4(time[inds4], f2 , t3, beta) + phi_0
    hplus[inds4]  = spherharm_norm*inc_plus*amp4*np.cos(phi4)/distance
    hcross[inds4] = spherharm_norm*inc_cross*amp4*np.sin(phi4)/distance

    return hplus, hcross

def nrpm_wrapper(freqs, params):

    kappa2T = lambda_2_kappa(params['mtot']/(1.+1./params['q']),
                             params['mtot']/(1.+ params['q']),
                             params['lambda1'], params['lambda2'])

    hp, hc = NRPM(params['srate'], params['seglen'], params['mtot'], params['q'], kappa2T,
                  params['distance'], params['iota'], params['phi_ref'])

    return hp, hc

def nrpm_extended_wrapper(freqs, params):

    kappa2T = lambda_2_kappa(params['mtot']/(1.+1./params['q']),
                             params['mtot']/(1.+ params['q']),
                             params['lambda1'], params['lambda2'])

    mtt = params['mtot']*MTSUN_SI
    hp, hc = NRPM(params['srate'], params['seglen'], params['mtot'], params['q'], kappa2T,
                  params['distance'], params['iota'], params['phi_ref'], f_merg = None,
                  alpha     = 1./params['NRPM_alpha_inverse']/mtt,
                  beta      = params['NRPM_beta']/mtt,
                  phi_kick  = params['NRPM_phi_pm'])

    return hp, hc

def nrpm_extended_recal_wrapper(freqs, params):

    kappa2T = lambda_2_kappa(params['mtot']/(1.+1./params['q']),
                             params['mtot']/(1.+ params['q']),
                             params['lambda1'], params['lambda2'])

    mtt = params['mtot']*MTSUN_SI
    hp, hc = NRPM(params['srate'], params['seglen'], params['mtot'], params['q'], kappa2T,
                  params['distance'], params['iota'], params['phi_ref'], f_merg = None,
                  alpha     = 1./params['NRPM_alpha_inverse']/mtt,
                  beta      = params['NRPM_beta']/mtt,
                  phi_kick  = params['NRPM_phi_pm'],
                  recal     = {ki : params[ki] for ki in params.keys() if 'recal' in ki})

    return hp, hc

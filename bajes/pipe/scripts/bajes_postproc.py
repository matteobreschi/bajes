#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals
import os
import numpy as np
import optparse as op

from scipy.special import logsumexp

try:
    import corner
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

from bajes                 import MSUN_SI, MTSUN_SI, CLIGHT_SI, GNEWTON_SI

from bajes.pipe            import ensure_dir, data_container, cart2sph, sph2cart, set_logger, save_dict_to_hdf5
from bajes.pipe.utils      import extract_snr

from bajes.obs.gw          import Detector, Noise, Series, Waveform
from bajes.obs.utils       import Cosmology
from bajes.obs.gw.utils    import compute_tidal_components, compute_lambda_tilde, compute_delta_lambda, mcq_to_m1, mcq_to_m2
from bajes.obs.gw.waveform import PolarizationTuple


def _stats(samples):
    md = np.median(samples)
    up = np.percentile(samples, 95)
    lo = np.percentile(samples, 5)
    return md, up-md, md-lo

def make_corner_plot(matrix, labels, outputname):

    N = len(labels)

    try:

        cornerfig=corner.corner(matrix,
                                labels          = labels,
                                bins            = 40,
                                color           = 'royalblue',
                                levels          = [.5, .9],
                                contour_kwargs  = {'colors':'navy','linewidths':0.95},
                                plot_datapoints = True,
                                show_titles     = True,
                                plot_density    = True,
                                smooth1d        = True,
                                smooth          = True)

        axes = np.array(cornerfig.axes).reshape((N,N))

        for i in range(N):
            ax = axes[i, i]
            ax.axvline(np.median(matrix[:,i]),         color="navy",        linestyle='--', linewidth  = 0.9)
            ax.axvline(np.percentile(matrix[:,i],5),   color="slateblue",   linestyle='--', linewidth  = 0.9)
            ax.axvline(np.percentile(matrix[:,i],95),  color="slateblue",   linestyle='--', linewidth  = 0.9)

        plt.savefig(outputname , dpi=100, bbox_inches='tight')
        plt.close()

    except Exception:
        logger.warning("Unable to produce corner plots, corner is not available. Please install corner.")

def make_corners(posterior, spin_flag, lambda_flag, ppdir, priors):

    # masses
    if not (('mtot' in priors.const) or ('mchirp' in priors.const) or ('q' in priors.const)):
        try:
            logger.info("... plotting masses ...")

            q_post  = posterior['q']
            nu_post = q_post/((1+q_post)*(1+q_post))
            if('mtot' in priors.names):
                mtot_post    = posterior['mtot']
                mtotq_matrix = np.column_stack((mtot_post, q_post))
                mtotq_labels = [r'$M_{tot} [{\rm M}_\odot]$',r'$q=m_1/m_2$']
                make_corner_plot(mtotq_matrix,  mtotq_labels, ppdir+'/mtotq_posterior.pdf')
            else:
                mchirp_post = posterior['mchirp']
                mcq_matrix  = np.column_stack((mchirp_post, q_post))
                mcq_labels  = [r'$M_{chirp} [{\rm M}_\odot]$',r'$q=m_1/m_2$']
                make_corner_plot(mcq_matrix,  mcq_labels, ppdir+'/mcq_posterior.pdf')

                mtot_post   = mchirp_post/np.power(np.abs(nu_post),3./5.)

            m1_post     = mtot_post/(1.+1./q_post)
            m2_post     = mtot_post/(1.+q_post)
            m1m2_matrix = np.column_stack((m1_post,m2_post))
            m1m2_labels = [r'$m_1 [{\rm M}_\odot]$',r'$m_2 [{\rm M}_\odot]$']
            make_corner_plot(m1m2_matrix, m1m2_labels,ppdir+'/m1m2_posterior.pdf')

        except(KeyError, ValueError):
            logger.info("Masses plot failed.")
    else:
        logger.info("Mass parameters were fixed. Skipping masses corner.")

    # spins
    if('align' in spin_flag):
        if not (('s1z' in priors.const) or ('s2z' in priors.const)):

            logger.info("... plotting spins ...")

            spin_matrix = np.column_stack((posterior['s1z'],posterior['s2z']))
            spin_labels = [r'$s_{1,z}$',r'$s_{2,z}$']
            make_corner_plot(spin_matrix,spin_labels,ppdir+'/spins_posterior.pdf')

            try:
                chieff_post = (m1_post * posterior['s1z'] + m2_post * posterior['s2z'])/mtot_post
                chiq_matrix = np.column_stack((chieff_post,q_post))
                chiq_labels = [r'$\chi_{eff}$',r'$q=m_1/m_2$']
                make_corner_plot(chiq_matrix,chiq_labels,ppdir+'/chiq_posterior.pdf')
            except Exception:
                logger.info("Aligned spins plot failed.")
        else:
            logger.info("Aligned spins parameters were fixed. Skipping aligned spins corner.")

    elif('precess' in spin_flag):

        logger.info("... plotting spins ...")
        spin_matrix = np.column_stack((posterior['s1'],posterior['tilt1']))
        spin_labels = [r'$s_{1}$',r'$\theta_{1L}$']
        make_corner_plot(spin_matrix,spin_labels,ppdir+'/spin1_posterior.pdf')

        spin_matrix = np.column_stack((posterior['s2'],posterior['tilt2']))
        spin_labels = [r'$s_{2}$',r'$\theta_{2L}$']
        make_corner_plot(spin_matrix,spin_labels,ppdir+'/spin2_posterior.pdf')

        try:
            chieff_post = (m1_post * posterior['s1'] * np.cos(posterior['tilt1']) + m2_post * posterior['s2'] * np.cos(posterior['tilt2']))/mtot_post
            chiq_matrix = np.column_stack((chieff_post,q_post))
            chiq_labels = [r'$\chi_{eff}$',r'$q=m_1/m_2$']
            make_corner_plot(chiq_matrix,chiq_labels,ppdir+'/chiq_posterior.pdf')
        except Exception:
            logger.info("Precessing spins chi_eff-q plot failed.")

        try:
            from bajes.obs.gw.utils import compute_chi_prec
            chip_post = np.array([compute_chi_prec(m1i,m2i,s1i,s2i,t1i,t2i) for m1i,m2i,s1i,s2i,t1i,t2i in zip(m1_post,m2_post,
                                                                                                               posterior['s1'],posterior['s2'],
                                                                                                               posterior['tilt1'],posterior['tilt2']) ])
            chichi_matrix = np.column_stack((chieff_post,chip_post))
            chiq_labels = [r'$\chi_{eff}$',r'$\chi_p$']
            make_corner_plot(chiq_matrix,chiq_labels,ppdir+'/chis_posterior.pdf')
        except Exception:
            logger.info("Precessing spins chi_eff-chip plot failed.")

    elif('no-spins' in spin_flag):
        logger.info("No spins option selected. Skipping spin plots.")

    else:
        logger.warning("Unknown spins option selected. Skipping spin plots.")

    # tides
    if not (('lambda1' in priors.const) or ('lambda2' in priors.const)):

        if(lambda_flag == 'bns-tides'):
            logger.info("... plotting tides ...")

            tide1_matrix = np.column_stack((posterior['lambda1'],posterior['lambda2']))
            tide1_labels = [r'$\Lambda_1$',r'$\Lambda_2$']
            make_corner_plot(tide1_matrix,tide1_labels,ppdir+'/tides_posterior.pdf')

            try:
                lambdat_post = compute_lambda_tilde(m1_post,m2_post,posterior['lambda1'],posterior['lambda2'])
                dlambda_post = compute_delta_lambda(m1_post,m2_post,posterior['lambda1'],posterior['lambda2'])

                tide2_matrix = np.column_stack((lambdat_post, dlambda_post))
                tide2_labels = [r'$\tilde \Lambda$', r'$\delta\tilde \Lambda$']
                make_corner_plot(tide2_matrix,tide2_labels,ppdir+'/lambdat_posterior.pdf')
            except Exception:
                logger.info("BNS-tides plot failed.")

        elif(lambda_flag == 'bhns-tides' or lambda_flag == 'nsbh-tides'):

            if(lambda_flag == 'nsbh-tides'):
                logger.warning("The 'nsbh-tides' string for the 'tidal-flag' option is deprecated and will be removed in a future release. Please use the 'nsbh-tides' string.")

            logger.info("... plotting tides ...")
            try:
                lambda1 = posterior['lambda1']
                lambda2 = 0.
                lambda_post = lambda1
            except KeyError:
                lambda1 = 0.
                lambda2 = posterior['lambda2']
                lambda_post = lambda2

            try:
                lambdat_post = compute_lambda_tilde(m1_post,m2_post,lambda1,lambda2)
                dlambda_post = compute_delta_lambda(m1_post,m2_post,lambda1,lambda2)

                tide_matrix = np.column_stack((lambda_post, lambdat_post,dlambda_post))
                tide_labels = [r'$\Lambda_{NS}$',r'$\tilde \Lambda$', r'$\delta\tilde \Lambda$']

            except Exception:

                tide_matrix = np.column_stack((lambda_post,np.zeros(len(lambda_post))))
                tide_labels = [r'$\Lambda_{NS}$', r'$\Lambda_{BH}$']

            make_corner_plot(tide_matrix,tide_labels,ppdir+'/lambdat_posterior.pdf')

        elif('no-tides' in lambda_flag):
            logger.info("No tides option selected. Skipping tides plots.")

        else:
            logger.warning("Unknown tides option selected. Skipping tides plots.")
    else:
        logger.info("Tides parameters were fixed. Skipping tides corner.")

    # sky location
    if not (('ra' in priors.const) or ('dec' in priors.const)):
        try:
            logger.info("... plotting sky location ...")
            skyloc_matrix = np.column_stack((posterior['ra'],posterior['dec']))
            skyloc_labels = [r'$\alpha [{\rm rad}]$', r'$\delta [{\rm rad}]$']
            make_corner_plot(skyloc_matrix,skyloc_labels,ppdir+'/skyloc_posterior.pdf')
        except Exception:
            logger.info("Sky position plot failed.")
    else:
        logger.info("Sky position parameters were fixed. Skipping sky position corner.")

    # distance - inclination
    if not (('distance' in priors.const) or ('cosi' in priors.const)):
        try:
            logger.info("... plotting distance-iota ...")
            iota_post = np.arccos(posterior['cosi'])
            distiot_matrix = np.column_stack((posterior['distance'], iota_post))
            distiot_labels = [r'$D_L [{\rm Mpc}]$', r'$\iota [{\rm rad}]$']
            make_corner_plot(distiot_matrix,distiot_labels,ppdir+'/distance_posterior.pdf')
        except Exception:
            logger.info("Distance-inclination plot failed.")
    else:
        logger.info("Distance-inclination parameters were fixed. Skipping distance-inclination corner.")

    # other
    if not (('psi' in priors.const) or ('phi_ref' in priors.const) or ('time_shift' in priors.const)):
        try:
            logger.info("... plotting external parameters ...")
            if('phi_ref' in priors.names):
                ext_matrix = np.column_stack((posterior['psi'],posterior['phi_ref'],posterior['time_shift']))
                ext_labels = [r'$\psi  [{\rm rad}]$', r'$\phi_{ref} [{\rm rad}]$', r'$t_0 [{\rm s}]$']
            else:
                ext_matrix = np.column_stack((posterior['psi'],posterior['time_shift']))
                ext_labels = [r'$\psi  [{\rm rad}]$', r'$t_0  [{\rm s}]$']
            make_corner_plot(ext_matrix,ext_labels,ppdir+'/external_posterior.pdf')
        except Exception:
            logger.info("External parameters plot failed.")
    else:
        logger.info("External parameters were fixed. Skipping external corner.")

    # other
    if(('energy' in priors.names) and ('angmom' in priors.names)):
        try:
            logger.info("... plotting hyperbolic parameters ...")
            ext_matrix = np.column_stack((posterior['energy'], posterior['angmom']))
            ext_labels = [r'$E_0/M$', r'$p_{\phi}^0$']
            make_corner_plot(ext_matrix,ext_labels,ppdir+'/hyperbolic_posterior.pdf')

            if(('align' in spin_flag) and not('s1z' in priors.const) and not('s2z' in priors.const)):
                logger.info("... plotting angular momentum parameters ...")
                ext_matrix = np.column_stack((posterior['s1z'], posterior['s2z'], posterior['angmom']))
                ext_labels = [r'$s_{1z}$', r'$s_{2z}$', r'$p_{\phi}^0$']
                make_corner_plot(ext_matrix,ext_labels,ppdir+'/angular_momentum_posterior.pdf')

        except Exception:
            logger.info("Hyperbolic parameters plot failed.")
    else:
        logger.info("Hyperbolic parameters were fixed or not included in the sampling. Skipping hyperbolic corner.")

def make_histograms(names, posterior_samples, prior_samples, outdir):

    from bajes.inf.utils import autocorrelation

    for i,ni in enumerate(names):

        logger.info("... producing histogram for {} ...".format(ni))

        try:
            mean, upper, lower  = _stats(posterior_samples[ni])

            fig = plt.figure()

            plt.title("{} = ".format(ni) + r"${"+ "{:.5f}".format(mean) + r"}^{ + "+ "{:.5f}".format(upper) + r"}_{ - "+ "{:.5f}".format(lower) +"}$")

            plt.hist(posterior_samples[ni], bins=66, edgecolor = 'royalblue', histtype='step', density=True, label="Posterior")
            plt.hist(prior_samples[ni],     bins=66, edgecolor = 'gray', histtype='step', density=True, label="Prior")

            plt.axvline(mean,   color='navy',ls='--', label="Median")
            plt.axvline(mean+upper,  color='slateblue',ls='--', label="90% C.L.")
            plt.axvline(mean-lower,  color='slateblue',ls='--')

            plt.ylabel('probability')
            plt.xlim((np.min(prior_samples[ni]), np.max(prior_samples[ni])))
            plt.xlabel('{}'.format(names[i]),size=12)
            plt.legend(loc='best')
            plt.savefig(outdir+'/hist_{}.pdf'.format(names[i]), dpi=100, bbox_inches='tight')

            plt.close()

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            ax1.set_title('{}'.format(names[i]),size=12)
            ax1.scatter(np.arange(len(posterior_samples[ni])), posterior_samples[ni], marker='.', color='royalblue')
            ax1.set_ylabel('samples')

            ax2.plot(autocorrelation(posterior_samples[ni]), lw=0.7, color='royalblue')
            ax2.set_ylabel('autocorr')
            ax2.set_xlabel('lag')

            plt.savefig(outdir+'/samples_{}.pdf'.format(names[i]), dpi=100, bbox_inches='tight')
            plt.close()

        except Exception:
            logger.warning("Unable to produce histogram plot for {}, an exception occurred.".format(ni))
            pass

    try:

        mean, upper, lower  = _stats(posterior_samples['logL'])
        logger.info("... producing histogram for logL ...")

        fig = plt.figure()

        plt.title("{} = ".format('logL') + r"${"+ "{:.5f}".format(mean) + r"}^{ + "+ "{:.5f}".format(upper) + r"}_{ - "+ "{:.5f}".format(lower) +"}$")

        plt.hist(posterior_samples['logL'], bins=66, edgecolor = 'royalblue', histtype='step', density=True, label="Likelihood")
        plt.hist(posterior_samples['logL']+posterior_samples['logPrior'], bins=66, edgecolor = 'gray', histtype='step', density=True, label="Posterior")

        plt.axvline(mean,   color='navy',ls='--', label="Median")
        plt.axvline(mean+upper,  color='slateblue',ls='--', label="90% C.L.")
        plt.axvline(mean-lower,  color='slateblue',ls='--')

        plt.ylabel('probability')
        plt.xlabel('logL',size=12)
        plt.legend(loc='best')
        plt.savefig(outdir+'/hist_{}.pdf'.format('logL'), dpi=100, bbox_inches='tight')

        plt.close()

    except KeyError:
        pass

def _wrap_compute_wf_and_snr(args):
    return compute_wf_and_snr(*args)

def compute_wf_and_snr(p, dets, noises, w, marg_phi=False, marg_time=False):

    # compute waveform
    hphc   = w.compute_hphc(p)

    # compute SNR
    phiref, tshift, snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det = extract_snr(list(dets.keys()), dets, hphc, p, w.domain, marg_phi=marg_phi, marg_time=marg_time)

    p['time_shift'] = p['time_shift'] + tshift
    hphc   = PolarizationTuple(plus  = hphc.plus*np.cos(phiref) - hphc.cross*np.sin(phiref),
                               cross = hphc.plus*np.sin(phiref) + hphc.cross*np.cos(phiref))

    # collect quantities
    wf = {}
    ww = {}
    sp = {}

    for det in dets.keys():

        h_tmp       = dets[det].project_tdwave(hphc, p, w.domain)
        h_strain    = Series('time', h_tmp, seglen=p['seglen'], srate=p['srate'], t_gps=p['t_gps'], f_min=p['f_min'], f_max=p['f_max'])
        sp[det]     = np.abs(h_strain.freq_series)

        # store waveform
        wf[det]     = h_strain.time_series

        # store whitened waveform
        h_strain.whitening(noises[det])
        ww[det]     = h_strain.time_series

    return snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det, sp, wf, ww

def reconstruct_waveform(outdir, posterior, container_inf, container_gw, N_samples=0, M_tot=None, pool=None):

    # get general information

    nsub_panels      = len(container_gw.datas.keys())
    strains_dets     = {det: {} for det in container_gw.datas.keys()}
    wfs              = {det: [] for det in container_gw.datas.keys()}
    wfw              = {det: [] for det in container_gw.datas.keys()}
    sps              = {det: [] for det in container_gw.datas.keys()}
    spd              = {det: [] for det in container_gw.datas.keys()}
    SNRs_mf_per_det  = {ifo: [] for ifo in container_inf.like.ifos}
    SNRs_opt_per_det = {ifo: [] for ifo in container_inf.like.ifos}
    SNRs_mf          = []
    SNRs_opt         = []
    data_output      = {}
    snr_output       = {}

    first_det                                 = list(container_gw.datas.keys())[0]
    data_first_det                            = container_gw.datas[first_det]
    freqs, f_min, f_max, t_gps, seglen, srate = data_first_det.freqs, data_first_det.f_min, data_first_det.f_max, data_first_det.t_gps, data_first_det.seglen, data_first_det.srate

    # initialize model
    names     = container_inf.prior.names
    constants = container_inf.prior.const
    w         = container_inf.like.wave

    # set sp-cal
    try:
        nspcal = container_inf.like.nspcal
    except Exception:
        # use exception for compatibility issues with old versions
        logger.warning("Unable to read information on spectral calibration envelopes. Setting nspcal = 0.")
        nspcal = 0

    if nspcal > 0 :
        spcal_freqs = np.logspace(1., np.log(np.max(freqs))/np.log(np.min(freqs)), base=np.min(freqs), num = nspcal)

    # estimate bandpassing frequency
    # set default in proximity of f_max
    f_max_bp = f_max-10
    if M_tot is not None:
        #Estimate of the ringdown frequency, approximating M_final with M_tot and using Schwarzschild value
        f_ringdown      = ((CLIGHT_SI**3)/(2.*np.pi*GNEWTON_SI*M_tot*MSUN_SI)) * (1.5251-1.1568*(1-0.7)**0.1292)
        temp_f_max_bp   = 2*f_ringdown
        # avoid f_max_bp to be larger than f_max
        if temp_f_max_bp < f_max_bp: f_max_bp = temp_f_max_bp
    logger.info("... bandpassing between [{:.0f}, {:.0f}] Hz ...".format(f_min,f_max_bp))

    # iterate on detectors
    from copy import deepcopy
    for det in strains_dets.keys():

        strains_dets[det]['s'] = container_gw.datas[det]
        strains_dets[det]['w'] = deepcopy(container_gw.datas[det])
        strains_dets[det]['d'] = container_gw.dets[det]
        strains_dets[det]['n'] = container_gw.noises[det]
        spd[det]               = strains_dets[det]['s'].freq_series

        if nspcal > 0 :
            strains_dets[det]['d'].store_measurement(strains_dets[det]['s'], strains_dets[det]['n'],
                                                     nspcal = nspcal, spcal_freqs = spcal_freqs)
        else:
            strains_dets[det]['d'].store_measurement(strains_dets[det]['s'], strains_dets[det]['n'])

        strains_dets[det]['w'].bandpassing(flow=f_min, fhigh=f_max_bp)
        strains_dets[det]['w'].whitening(strains_dets[det]['n'])

    if ((N_samples==0) or (N_samples > len(posterior))):    samples_list = np.arange(0,len(posterior))
    else:                                                   samples_list = np.random.choice(len(posterior), N_samples, replace=False)

    logger.info("... extracting {} posterior samples ...".format(len(samples_list)))
    if pool is None:

        for j,k in enumerate(samples_list):

            # generate waveform
            params = {name: posterior[name][k] for name in names}
            p      = {**params,**constants}

            # compute waveform and snr
            snr_mf, snr_mf_per_det, snr_opt, snr_opt_per_det, _sp, _wf, _ww = compute_wf_and_snr(p,
                                                                                                 {ifo: strains_dets[ifo]['d'] for ifo in container_inf.like.ifos},
                                                                                                 {ifo: strains_dets[ifo]['n'] for ifo in container_inf.like.ifos},
                                                                                                 w, 
                                                                                                 marg_phi=container_inf.like.marg_phi_ref,
                                                                                                 marg_time=container_inf.like.marg_time_shift)

            # collect quantities
            SNRs_mf.append(snr_mf)
            SNRs_opt.append(snr_opt)

            for det in strains_dets.keys():
                SNRs_mf_per_det[det].append(snr_mf_per_det[det])
                SNRs_opt_per_det[det].append(snr_opt_per_det[det])
                sps[det].append(_sp[det])
                wfs[det].append(_wf[det])
                wfw[det].append(_ww[det])

    else:

        from itertools import repeat

        # list all params
        params  = [ {**{name: posterior[name][k] for name in names}, **constants} for k in samples_list ]
        dets    = {ifo: strains_dets[ifo]['d'] for ifo in container_inf.like.ifos}
        nois    = {ifo: strains_dets[ifo]['n'] for ifo in container_inf.like.ifos}

        # compute with pool
        results = list(pool.map(_wrap_compute_wf_and_snr,
                                zip(params, repeat(dets), repeat(nois), repeat(w),
                                    repeat(container_inf.like.marg_phi_ref),
                                    repeat(container_inf.like.marg_time_shift))))


        # unpack
        SNRs_mf  = [r[0] for r in results]
        SNRs_opt = [r[2] for r in results]
        for det in strains_dets.keys():
            SNRs_mf_per_det[det]  = [r[1][det] for r in results]
            SNRs_opt_per_det[det] = [r[3][det] for r in results]
            sps[det]              = [r[4][det] for r in results]
            wfs[det]              = [r[5][det] for r in results]
            wfw[det]             = [r[6][det] for r in results]

        del results

    # print and plot recovered SNRs
    snr_output['indices']     = samples_list
    snr_output['network_mf']  = SNRs_mf
    snr_output['network_opt'] = SNRs_opt
    for ifo in container_inf.like.ifos:
        snr_output[ifo+'_mf']  = SNRs_mf_per_det[ifo]
        snr_output[ifo+'_opt'] = SNRs_opt_per_det[ifo]
        logger.info(" > Recovered {} SNR  (mf) = {:.3f} + {:.3f} - {:.3f}".format(ifo, *_stats(SNRs_mf_per_det[ifo])))
        logger.info(" > Recovered {} SNR (opt) = {:.3f} + {:.3f} - {:.3f}".format(ifo, *_stats(SNRs_opt_per_det[ifo])))
    logger.info(" > Recovered Network SNR  (mf) = {:.3f} + {:.3f} - {:.3f}".format(*_stats(SNRs_mf)))
    logger.info(" > Recovered Network SNR (opt) = {:.3f} + {:.3f} - {:.3f}".format(*_stats(SNRs_opt)))

    snr_matrix_mf = np.column_stack([SNRs_mf]+[SNRs_mf_per_det[ifo] for ifo in container_inf.like.ifos])
    snr_labels_mf = [r"${\rm Net.}$ ${\rm SNR_{mf}}$"] + [r"${\rm " + ifo + r"}$ ${\rm SNR_{mf}}$" for ifo in container_inf.like.ifos]
    make_corner_plot(snr_matrix_mf,  snr_labels_mf, outdir +'/../snr/corner_mf.pdf')

    snr_matrix_opt = np.column_stack([SNRs_opt]+[SNRs_opt_per_det[ifo] for ifo in container_inf.like.ifos])
    snr_labels_opt = [r"${\rm Net.}$ ${\rm SNR_{opt}}$"] + [r"${\rm " + ifo + r"}$ ${\rm SNR_{opt}}$" for ifo in container_inf.like.ifos]
    make_corner_plot(snr_matrix_opt,  snr_labels_opt, outdir +'/../snr/corner_opt.pdf')

    snr_mf_fl = open(outdir +'/../snr/posterior_mf.dat'.format(det),'w')
    snr_mf_fl.write('#\t index \t network SNR (mf)' + ''.join([' \t ' + ifo for ifo in container_inf.like.ifos] + [' \n']))
    for i in range(len(SNRs_mf)):
        snr_mf_fl.write("{} {} ".format(samples_list[i], SNRs_mf[i]))
        for ifo in container_inf.like.ifos:
            snr_mf_fl.write("\t {} ".format(SNRs_mf_per_det[ifo][i]))
        snr_mf_fl.write("\n")
    snr_mf_fl.close()

    snr_opt_fl = open(outdir +'/../snr/posterior_opt.dat'.format(det),'w')
    snr_opt_fl.write('#\t index \t network SNR (opt)' + ''.join([' \t ' + ifo for ifo in container_inf.like.ifos] + [' \n']))
    for i in range(len(SNRs_opt)):
        snr_opt_fl.write("{} {} ".format(samples_list[i], SNRs_opt[i]))
        for ifo in container_inf.like.ifos:
            snr_opt_fl.write("\t {} ".format(SNRs_opt_per_det[ifo][i]))
        snr_opt_fl.write("\n")
    snr_opt_fl.close()

    # Plot the data
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(hspace = .001)

    data_output['strain']   = {}
    data_output['noise']    = {}
    data_output['waveform'] = {}

    # plot median, upper, lower and save waveform
    for i,det in enumerate(strains_dets.keys()):

        data_output['strain'][det] = {}
        data_output['noise'][det] = {}
        data_output['waveform'][det] = {}

        lo, me, hi = np.percentile(wfs[det],[5,50,95], axis=0)

        ax = fig.add_subplot(nsub_panels,1,i+1)

        ax.plot(strains_dets[det]['s'].times-t_gps, strains_dets[det]['s'].time_series, c='gray', lw=.5, label='Data')
        ax.plot(strains_dets[det]['s'].times-t_gps, me, c='navy', lw=0.8, label='Waveform')
        ax.fill_between(strains_dets[det]['s'].times-t_gps, lo, hi, color='navy', alpha=0.4, lw=0.5)

        data_output['strain'][det]['time']        = strains_dets[det]['s'].times
        data_output['strain'][det]['series']      = strains_dets[det]['s'].time_series
        data_output['waveform'][det]['time']      = strains_dets[det]['s'].times
        data_output['waveform'][det]['series']    = me
        data_output['waveform'][det]['series_up'] = hi
        data_output['waveform'][det]['series_lo'] = lo

        ax.set_xlabel('t - t$_{\mathrm{gps}}$ [s]')
        ax.set_ylabel('Strain {}'.format(det))
        ax.set_xlim((-seglen/4,seglen/4))
        if i==0: ax.legend(loc='upper right', prop={'size': 10})
        if not(i==len(strains_dets.keys())-1):
            ax.get_xaxis().set_visible(False)

        wf_ci_fl = open(outdir +'/waveform_{}.dat'.format(det),'w')
        wf_ci_fl.write('#\t time \t median \t lower \t higher\n')
        for i in range(len(strains_dets[det]['s'].times)):
            wf_ci_fl.write("%.10f \t %.10g \t %.10g \t %.10g \n" %(strains_dets[det]['s'].times[i], me[i], lo[i], hi[i]))
        wf_ci_fl.close()

    plt.savefig(outdir +'/Reconstructed_waveform.pdf', bbox_inches='tight', dpi=100)
    plt.close()

    # Plot the data
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(hspace = .001)

    # plot median, upper, lower and save whitened waveform
    for i,det in enumerate(strains_dets.keys()):

        lo, me, hi = np.percentile(wfw[det],[5,50,95], axis=0)

        ax = fig.add_subplot(nsub_panels,1,i+1)

        ax.plot(strains_dets[det]['w'].times-t_gps, strains_dets[det]['w'].time_series, c='gray', lw=.5, label='Data')
        ax.plot(strains_dets[det]['w'].times-t_gps, me, c='navy', lw=0.8, label='Waveform')
        ax.fill_between(strains_dets[det]['s'].times-t_gps, lo, hi, color='navy', alpha=0.4, lw=0.5)

        data_output['strain'][det]['series_whiten']      = strains_dets[det]['w'].time_series
        data_output['waveform'][det]['series_whiten']    = me
        data_output['waveform'][det]['series_whiten_up'] = hi
        data_output['waveform'][det]['series_whiten_lo'] = lo

        ax.set_xlabel('t - t$_{\mathrm{gps}}$ [s]')
        ax.set_ylabel('Strain {}'.format(det))
        ax.set_xlim((-seglen/4,seglen/4))
        if i==0: ax.legend(loc='upper right', prop={'size': 10})
        if not(i==len(strains_dets.keys())-1):
            ax.get_xaxis().set_visible(False)

        wf_ci_fl = open(outdir +'/whiten_waveform_{}.dat'.format(det),'w')
        wf_ci_fl.write('#\t time \t median \t lower \t higher\n')
        for i in range(len(strains_dets[det]['w'].times)):
            wf_ci_fl.write("%.10f \t %.10g \t %.10g \t %.10g \n" %(strains_dets[det]['w'].times[i], me[i], lo[i], hi[i]))
        wf_ci_fl.close()

    plt.savefig(outdir +'/Reconstructed_waveform_whitened.pdf', bbox_inches='tight', dpi=100)
    plt.close()

    if not(M_tot==None):
        # Repeating the above plot while zooming on the merger.
        # FIXME: this could be done in a single shot, storing the axes.

        # waveform
        fig = plt.figure(figsize=(8,6))
        plt.subplots_adjust(hspace = .001)

        t_peak = None
        for i,det in enumerate(strains_dets.keys()):

            lo, me, hi = np.percentile(wfs[det],[5,50,95], axis=0)

            ax = fig.add_subplot(nsub_panels,1,i+1)

            # Compute the t_peak of the median waveform, relative to the gps time
            if t_peak is None : t_peak = strains_dets[det]['s'].times[np.argmax(np.abs(me))]-t_gps

            ax.plot(strains_dets[det]['s'].times-t_gps, strains_dets[det]['s'].time_series, c='k', lw=0.8, label='Data')
            ax.plot(strains_dets[det]['s'].times-t_gps, me, c='royalblue', lw=0.8, label='Waveform')
            ax.fill_between(strains_dets[det]['s'].times-t_gps, lo, hi, color='royalblue', alpha=0.4, lw=0.5)

            ax.set_xlim([t_peak-200*M_tot*MTSUN_SI, t_peak+200*M_tot*MTSUN_SI])
            ax.set_xlabel('t - t$_{\mathrm{gps}}$ [s]')
            ax.set_ylabel('Strain {}'.format(det))
            if i==0: ax.legend(loc='upper right', prop={'size': 10})
            if not(i==len(strains_dets.keys())-1):
                ax.get_xaxis().set_visible(False)

        plt.savefig(os.path.join(outdir, 'Reconstructed_waveform_zoom.pdf'), bbox_inches='tight', dpi=100)
        plt.close()

        # whiten waveform
        fig = plt.figure(figsize=(8,6))
        plt.subplots_adjust(hspace = .001)

        t_peak = None
        for i,det in enumerate(strains_dets.keys()):

            lo, me, hi = np.percentile(wfw[det],[5,50,95], axis=0)

            ax = fig.add_subplot(nsub_panels,1,i+1)

            # Compute the t_peak of the median waveform, relative to the gps time
            if t_peak is None : t_peak = strains_dets[det]['s'].times[np.argmax(np.abs(me))]-t_gps

            ax.plot(strains_dets[det]['w'].times-t_gps, strains_dets[det]['w'].time_series, c='k', lw=0.8, label='Data')
            ax.plot(strains_dets[det]['w'].times-t_gps, me, c='royalblue', lw=0.8, label='Waveform')
            ax.fill_between(strains_dets[det]['s'].times-t_gps, lo, hi, color='royalblue', alpha=0.4, lw=0.5)

            ax.set_xlim([t_peak-200*M_tot*MTSUN_SI, t_peak+200*M_tot*MTSUN_SI])
            ax.set_xlabel('t - t$_{\mathrm{gps}}$ [s]')
            ax.set_ylabel('Strain {}'.format(det))
            if i==0: ax.legend(loc='upper right', prop={'size': 10})
            if not(i==len(strains_dets.keys())-1):
                ax.get_xaxis().set_visible(False)

        plt.savefig(os.path.join(outdir, 'Reconstructed_waveform_whitened_zoom.pdf'), bbox_inches='tight', dpi=100)
        plt.close()

    # plot median, upper, lower and save spectrum
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(hspace = .001)

    for i,det in enumerate(strains_dets.keys()):

        lo, me, hi = np.percentile(sps[det],[5,50,95], axis=0)

        ax = fig.add_subplot(nsub_panels,1,i+1)

        ax.loglog(strains_dets[det]['s'].freqs, np.abs(spd[det]), c='gray', lw=.1, label='Data', zorder=0)
        ax.loglog(strains_dets[det]['s'].freqs, me, c='royalblue', lw=0.8, label='Waveform')
        ax.fill_between(strains_dets[det]['s'].freqs, lo, hi, color='royalblue', alpha=0.4, lw=0.5)
        ax.loglog(strains_dets[det]['n'].freqs, strains_dets[det]['n'].amp_spectrum*np.sqrt(seglen), c='navy', lw=1, label='ASD')

        data_output['noise'][det]['freq']           = strains_dets[det]['n'].freqs
        data_output['noise'][det]['asd']            = strains_dets[det]['n'].amp_spectrum
        data_output['strain'][det]['freq']          = strains_dets[det]['s'].freqs
        data_output['strain'][det]['spectrum']      = np.abs(spd[det])
        data_output['waveform'][det]['freq']        = strains_dets[det]['s'].freqs
        data_output['waveform'][det]['spectrum']    = me
        data_output['waveform'][det]['spectrum_up'] = hi
        data_output['waveform'][det]['spectrum_lo'] = lo

        bucket = np.min(strains_dets[det]['n'].amp_spectrum*np.sqrt(seglen))
        ax.set_xlabel(r'$f$ [Hz]')
        ax.set_ylabel('Spectrum {}'.format(det))
        ax.set_xlim((f_min, f_max))
        ax.set_ylim((bucket*1e-3,bucket*1e2))
        if i==0: ax.legend(loc='upper right', prop={'size': 10})
        if not(i==len(strains_dets.keys())-1):
            ax.get_xaxis().set_visible(False)

        wf_ci_fl = open(outdir +'/spectrum_{}.dat'.format(det),'w')
        wf_ci_fl.write('#\t freq \t median \t lower \t higher\n')
        inds = np.where((strains_dets[det]['s'].freqs>=f_min)&(strains_dets[det]['s'].freqs<=f_max))[0]
        for i in inds:
            wf_ci_fl.write("%.10f \t %.10g \t %.10g \t %.10g \n" %(strains_dets[det]['s'].freqs[i], me[i], lo[i], hi[i]))
        wf_ci_fl.close()

    plt.savefig(outdir +'/Reconstructed_spectrum.pdf', bbox_inches='tight', dpi=100)
    plt.close()

    return data_output, snr_output

def compute_cosmology_and_masses(names, cosmo, posterior_samples, prior_samples, map_fn):

    cosmo  = Cosmology(cosmo=cosmo, kwargs=None)
    post_dict = {}
    prio_dict = {}

    if 'distance' in names:

        post_dict['z'] = np.array(list(map_fn(cosmo.dl_to_z, posterior_samples['distance'])))
        prio_dict['z'] = np.array(list(map_fn(cosmo.dl_to_z, prior_samples['distance'])))

        if 'mtot' in names:
            post_dict['mtot_src'] = posterior_samples['mtot']/(1+post_dict['z'])
            prio_dict['mtot_src'] = prior_samples['mtot']/(1+prio_dict['z'])

        elif 'mchirp' in names:
            post_dict['mchirp_src'] = posterior_samples['mchirp']/(1+post_dict['z'])
            prio_dict['mchirp_src'] = prior_samples['mchirp']/(1+prio_dict['z'])

        else:
            logger.warning("Unable to compute source-frame mass from posterior samples. Skipping this quantity.")

    else:
        logger.warning("Unable to compute redshift from posterior samples. Skipping this quantity.")

    return post_dict, prio_dict

def make_final_summary(outdir,
                       posterior_samples, prior_samples,
                       z_posterior_samples, z_prior_samples,
                       container_inf, container_gw,
                       data_dict, snr_dict,
                       nprior=None):

    # get information
    names            = container_inf.prior.names
    bounds           = container_inf.prior.bounds
    consts           = container_inf.prior.const
    consts['approx'] = container_inf.like.wave.approx

    # set sp-cal
    try:
        consts['nspcal'] = container_inf.like.nspcal
    except Exception:
        # use exception for compatibility issues with old versions
        logger.warning("Unable to read information on spectral calibration envelopes. Setting nspcal = 0.")
        consts['nspcal'] = 0

    # set psd-weight
    try:
        consts['nweights'] = container_inf.like.nweights
    except Exception:
        # use exception for compatibility issues with old versions
        logger.warning("Unable to read information on PSD weights. Setting nweights = 0.")
        consts['nweights'] = 0

    # posterior and prior samples
    pri     = {ni: np.array(prior_samples[ni]) for i,ni in enumerate(names)}
    pos     = {ni: np.array(posterior_samples[ni]) for ni in np.append(names , ['logL', 'logPrior'])}
    z_pri   = {ni: np.array(z_prior_samples[ni]) for ni in list(z_prior_samples.keys())}
    z_pos   = {ni: np.array(z_posterior_samples[ni]) for ni in list(z_prior_samples.keys())}

    # clean data
    snr_dict = {ki : np.array(snr_dict[ki]) for ki in snr_dict.keys()}

    # define final output
    # TODO: include bayes_factor, sampler information
    summary = {'parameters':        {'names': np.array(names),
                                     'bounds': np.array(bounds)},
               'constants':         consts,
               'data':              data_dict,
               'prior_samples':     {**pri, **z_pri},
               'posterior_samples': {**pos, **z_pos},
               'snr':               snr_dict }

    # save hdf5
    save_dict_to_hdf5(summary, 'summary/', os.path.join(outdir,'../summary.hdf5'))

def clean_outdir(outdir):

    # list folders
    listdir = os.listdir(outdir)

    # making folder for pickles
    run_dir = os.path.abspath(outdir+'/run')
    if os.path.exists(run_dir):
        pkl_dir = os.path.abspath(run_dir+'/pkl')
    else:
        pkl_dir = os.path.abspath(outdir+'/pkl')
    ensure_dir(pkl_dir)

    for di in listdir:
        if di.split('.')[-1] == 'pkl' and os.path.isfile(outdir+'/'+di):
            os.replace(outdir+'/'+di, pkl_dir+'/'+di)


if __name__ == "__main__":

    parser=op.OptionParser()

    # Required options
    parser.add_option('-o','--outdir',    dest='outdir',          default=None,           type='string',                        help="Name of the directory containing the output of the run.")

    # Optional options
    parser.add_option('-n', '--nprocs',   dest='nprocs',          default=0,              type='int',                           help='Optional: Number of parallel threads. Dafault: 0 (serial)')
    parser.add_option('-v', '--verbose',  dest='silence',         default=True,           action="store_false",                 help='Optional: Activate stream handler, use this if you are running on terminal. Dafault: False')
    parser.add_option('--M-tot-estimate', dest='M_tot',           default='posterior',                                          help="Optional: Estimate of the total mass of the system, if not None, it is used to set narrower bandpassing and merger zoom. If equal to 'posterior', the value is extracted from the posterior samples. If a float is passed, that value is used instead. Default: None.")
    parser.add_option('--N-samples-wf',   dest='N_samples_wf',    default=3000,           type='int',                           help="Optional: Number of samples to be used in waveform reconstruction. If 0, all samples are used. Default: 3000.")
    parser.add_option('--N-samples-prior',dest='nprior',          default=None,           type='int',                           help="Optional: Number of prior samples. Default: Min( Npost, 10000 ).")
    parser.add_option('--spin-flag',      dest='spin_flag',       default='no-spins',     type='string',                        help="Optional: Spin prior flag. Default: 'no-spins'. Available options: ['no-spins', 'align', 'precess'].")
    parser.add_option('--tidal-flag',     dest='lambda_flag',     default='no-tides',     type='string',                        help="Optional: Spin prior flag. Default: 'no-tides'. Available options: ['no-tides', 'bns-tides', 'bhns-tides'].")
    parser.add_option('--cosmo',          dest='cosmo',           default='Planck18_arXiv_v2',     type='string',               help="Optional: Cosmology model for redshift computation. Default: Planck18_arXiv_v2")
    parser.add_option('--seed',           dest='seed',            default=None,           type='int',                           help="Optional: Seed for pseudo-random number generator.")
    (opts,args) = parser.parse_args()

    if opts.outdir is not None: outdir = opts.outdir
    else: raise ValueError("The 'outdir' option is required. Aborting.")
    if opts.seed is not None: np.random.seed(opts.seed)

    ppdir       = os.path.abspath(outdir+'/postproc')
    wf_dir      = os.path.abspath(os.path.join(ppdir, 'waveform'))
    hist_dir    = os.path.abspath(os.path.join(ppdir, 'hist'))
    corner_dir  = os.path.abspath(os.path.join(ppdir, 'corner'))
    snr_dir  = os.path.abspath(os.path.join(ppdir, 'snr'))

    ensure_dir(ppdir)
    ensure_dir(corner_dir)
    ensure_dir(hist_dir)
    ensure_dir(wf_dir)
    ensure_dir(snr_dir)

    global logger

    logger = set_logger(outdir=ppdir, label='bajes_postproc', silence=opts.silence)
    logger.info("Running bajes postprocessing:")
    logger.info("The reported uncertainties correpond to 90% credible regions.")
    logger.info("The contours of the corner plots represent 50%, 90% credible regions.")

    run_dir_output = os.path.join(outdir, 'run')

    # extract posterior and prior object from pickle
    if not(os.path.exists(run_dir_output)): 
        posterior = np.genfromtxt( os.path.join(outdir, 'posterior.dat'),     names=True)
        pkl_dir_output = os.path.join(outdir, 'pkl')

    else:                                   
        posterior = np.genfromtxt( os.path.join(outdir, 'run/posterior.dat'), names=True)
        pkl_dir_output = os.path.join(run_dir_output, 'pkl')

    if os.path.exists(pkl_dir_output):
        dc        = data_container(os.path.join(pkl_dir_output, 'inf.pkl'))
        dc_gw     = data_container(os.path.join(pkl_dir_output, 'gw_obs.pkl'))
    elif os.path.exists(run_dir_output):
        dc        = data_container(os.path.join(run_dir_output, 'inf.pkl'))
        dc_gw     = data_container(os.path.join(run_dir_output, 'gw_obs.pkl'))
    else:
        dc        = data_container(os.path.join(outdir, 'inf.pkl'))
        dc_gw     = data_container(os.path.join(outdir, 'gw_obs.pkl'))

    container_inf = dc.load()
    container_gw  = dc_gw.load()

    del dc, dc_gw
    priors        = container_inf.prior

    # set pool
    if opts.nprocs <= 1 :
        pool = None
        map_fn = map
    else:
        from bajes.pipe import initialize_mthr_pool
        pool, close_pool   = initialize_mthr_pool(opts.nprocs)
        map_fn = pool.map

    # extract prior samples
    logger.info("Extracting prior samples...")
    if opts.nprior is None: opts.nprior = min(len(posterior), 10000)
    prior_samples = np.transpose(list(map_fn(priors.get_prior_samples, np.ones(opts.nprior, dtype=int))))
    prior_samples = {ni: prior_samples[i][0] for i,ni in enumerate(priors.names)}

    # compute redshift
    logger.info("Computing redshifts...")
    z_post, z_prio = compute_cosmology_and_masses(priors.names, opts.cosmo, posterior, prior_samples, map_fn)

    # produce histogram plots
    logger.info("Producing histograms...")
    make_histograms(priors.names, posterior, prior_samples, hist_dir)
    make_histograms(list(z_post.keys()), z_post, z_prio, hist_dir)

    # produce corner plots
    logger.info("Producing corners...")
    make_corners(posterior, opts.spin_flag, opts.lambda_flag, corner_dir, priors)

    # get Mtot estimate
    if not(opts.M_tot==None):
        if(opts.M_tot=='posterior'):
            if('mtot' in priors.names):
                opts.M_tot = np.median(posterior['mtot'])
            elif('mtot' in priors.const):
                opts.M_tot = priors.const['mtot']
            elif(('mchirp' in priors.names) and ('q' in priors.names)):
                opts.M_tot = mcq_to_m1(np.median(posterior['mchirp']), np.median(posterior['q'])) + mcq_to_m2(np.median(posterior['mchirp']), np.median(posterior['q']))
            elif(('mchirp' in priors.names) and ('q' in priors.const)):
                opts.M_tot = mcq_to_m1(np.median(posterior['mchirp']), priors.const['q']) + mcq_to_m2(np.median(posterior['mchirp']), priors.const['q'])
            elif(('mchirp' in priors.const) and ('q' in priors.names)):
                opts.M_tot = mcq_to_m1(priors.const['mchirp'], np.median(posterior['q'])) + mcq_to_m2(priors.const['mchirp'], np.median(posterior['q']))
            elif(('mchirp' in priors.const) and ('q' in priors.const)):
                opts.M_tot = mcq_to_m1(priors.const['mchirp'], priors.const['q']) + mcq_to_m2(priors.const['mchirp'], priors.const['q'])
            else:
                logger.warning("Could not extract M_tot (either directly or through related mass parameters) from posterior or fixed parameters. Setting it to None and skipping zoomed plots.")
                opts.M_tot = None
        else:
            opts.M_tot = float(opts.M_tot)

    # produce waveform plots
    logger.info("Reconstructing waveforms...")
    if pool is None :
        data_dict, snr_dict = reconstruct_waveform(wf_dir, posterior, container_inf, container_gw, N_samples = opts.N_samples_wf, M_tot = opts.M_tot)
    else:
        data_dict, snr_dict = reconstruct_waveform(wf_dir, posterior, container_inf, container_gw, N_samples = opts.N_samples_wf, M_tot = opts.M_tot, pool=pool)
        close_pool(pool)

    # generate final summary
    logger.info("Generating hdf5 summary...")
    make_final_summary(ppdir, posterior, prior_samples, z_post, z_prio, container_inf, container_gw, data_dict, snr_dict)

    # clean outdir
    clean_outdir(outdir)

    logger.info("... done.")

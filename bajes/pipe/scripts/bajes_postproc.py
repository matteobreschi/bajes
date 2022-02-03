#!/usr/bin/env python
from __future__ import division, unicode_literals
import os
import numpy as np
import optparse as op

try:
    import corner
except Exception:
    pass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass

from bajes.pipe            import ensure_dir, data_container, cart2sph, sph2cart, set_logger
from bajes.obs.gw          import Detector, Noise, Series, Waveform
from bajes.obs.gw.utils    import compute_tidal_components, compute_lambda_tilde, compute_delta_lambda, mcq_to_m1, mcq_to_m2
from bajes.obs.gw.waveform import PolarizationTuple
from bajes import MSUN_SI, MTSUN_SI, CLIGHT_SI, GNEWTON_SI

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

        plt.savefig(outputname , dpi=250)
        plt.close()

    except Exception:
        logger.warning("Unable to produce corner plots, corner is not available. Please install corner.")

def make_corners(posterior, spin_flag, lambda_flag, extra_flag, ppdir):

    # masses
    try:
        logger.info("... plotting masses ...")
        mchirp_post   = posterior['mchirp']
        q_post      = posterior['q']

        nu_post     = q_post/((1+q_post)*(1+q_post))
        mtot_post   = mchirp_post/np.power(np.abs(nu_post),3./5.)

        m1_post     = mtot_post/(1.+1./q_post)
        m2_post     = mtot_post/(1.+q_post)

        m1m2_matrix = np.column_stack((m1_post,m2_post))
        mcq_matrix  = np.column_stack((mchirp_post,q_post))

        m1m2_labels = [r'$m_1 [{\rm M}_\odot]$',r'$m_2 [{\rm M}_\odot]$']
        mcq_labels  = [r'$M_{chirp} [{\rm M}_\odot]$',r'$q=m_1/m_2$']

        make_corner_plot(m1m2_matrix,m1m2_labels,ppdir+'/m1m2_posterior.png')
        make_corner_plot(mcq_matrix,mcq_labels,ppdir+'/mcq_posterior.png')
    except(KeyError, ValueError):
        pass

    # spins
    if('align' in spin_flag):
        logger.info("... plotting spins ...")

        spin_matrix = np.column_stack((posterior['s1z'],posterior['s2z']))
        spin_labels = [r'$s_{1,z}$',r'$s_{2,z}$']
        make_corner_plot(spin_matrix,spin_labels,ppdir+'/spins_posterior.png')

        try:
            chieff_post = (m1_post * posterior['s1z'] + m2_post * posterior['s2z'])/mtot_post
            chiq_matrix = np.column_stack((chieff_post,q_post))
            chiq_labels = [r'$\chi_{eff}$',r'$q=m_1/m_2$']
            make_corner_plot(chiq_matrix,chiq_labels,ppdir+'/chiq_posterior.png')
        except Exception:
            pass

    elif('precess' in spin_flag):

        logger.info("... plotting spins ...")
        spin_matrix = np.column_stack((posterior['s1'],posterior['tilt1']))
        spin_labels = [r'$s_{1}$',r'$\theta_{1L}$']
        make_corner_plot(spin_matrix,spin_labels,ppdir+'/spin1_posterior.png')

        spin_matrix = np.column_stack((posterior['s2'],posterior['tilt2']))
        spin_labels = [r'$s_{2}$',r'$\theta_{2L}$']
        make_corner_plot(spin_matrix,spin_labels,ppdir+'/spin2_posterior.png')

        try:
            chieff_post = (m1_post * posterior['s1'] * np.cos(posterior['tilt1']) + m2_post * posterior['s2'] * np.cos(posterior['tilt2']))/mtot_post
            chiq_matrix = np.column_stack((chieff_post,q_post))
            chiq_labels = [r'$\chi_{eff}$',r'$q=m_1/m_2$']
            make_corner_plot(chiq_matrix,chiq_labels,ppdir+'/chiq_posterior.png')
        except Exception:
            pass

        try:
            from bajes.obs.gw.utils import compute_chi_prec
            chip_post = np.array([compute_chi_prec(m1i,m2i,s1i,s2i,t1i,t2i) for m1i,m2i,s1i,s2i,t1i,t2i in zip(m1_post,m2_post,
                                                                                                               posterior['s1'],posterior['s2'],
                                                                                                               posterior['tilt1'],posterior['tilt2']) ])
            chichi_matrix = np.column_stack((chieff_post,chip_post))
            chiq_labels = [r'$\chi_{eff}$',r'$\chi_p$']
            make_corner_plot(chiq_matrix,chiq_labels,ppdir+'/chis_posterior.png')
        except Exception:
            pass

    elif('no-spins' in spin_flag):
        logger.info("No spins option selected. Skipping spin plots.")

    else:
        logger.warning("Unknown spins option selected. Skipping spin plots.")

    # tides
    if(lambda_flag == 'bns-tides'):
        logger.info("... plotting tides ...")

        tide1_matrix = np.column_stack((posterior['lambda1'],posterior['lambda2']))
        tide1_labels = [r'$\Lambda_1$',r'$\Lambda_2$']
        make_corner_plot(tide1_matrix,tide1_labels,ppdir+'/tides_posterior.png')

        try:
            lambdat_post = compute_lambda_tilde(m1_post,m2_post,posterior['lambda1'],posterior['lambda2'])
            dlambda_post = compute_delta_lambda(m1_post,m2_post,posterior['lambda1'],posterior['lambda2'])

            tide2_matrix = np.column_stack((lambdat_post, dlambda_post))
            tide2_labels = [r'$\tilde \Lambda$', r'$\delta\tilde \Lambda$']
            make_corner_plot(tide2_matrix,tide2_labels,ppdir+'/lambdat_posterior.png')
        except Exception:
            pass

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

        make_corner_plot(tide_matrix,tide_labels,ppdir+'/lambdat_posterior.png')

    elif('no-tides' in lambda_flag):
        logger.info("No spins option selected. Skipping tides plots.")

    else:
        logger.warning("Unknown tides option selected. Skipping tides plots.")

    # sky location
    try:
        logger.info("... plotting sky location ...")
        skyloc_matrix = np.column_stack((posterior['ra'],posterior['dec']))
        skyloc_labels = [r'$\alpha [{\rm rad}]$', r'$\delta [{\rm rad}]$']
        make_corner_plot(skyloc_matrix,skyloc_labels,ppdir+'/skyloc_posterior.png')
    except Exception:
        pass

    # distance - inclination
    try:
        logger.info("... plotting distance-iota ...")
        iota_post = np.arccos(posterior['cosi'])
        distiot_matrix = np.column_stack((posterior['distance'], iota_post))
        distiot_labels = [r'$D_L [{\rm Mpc}]$', r'$\iota [{\rm rad}]$']
        make_corner_plot(distiot_matrix,distiot_labels,ppdir+'/distance_posterior.png')
    except Exception:
        pass

    # other
    try:
        logger.info("... plotting external parameters ...")
        try:
            ext_matrix = np.column_stack((posterior['psi'],posterior['phi_ref'],posterior['time_shift']))
            ext_labels = [r'$\psi  [{\rm rad}]$', r'$\phi_{ref} [{\rm rad}]$', r'$t_0 [{\rm s}]$']
        except Exception:
            ext_matrix = np.column_stack((posterior['psi'],posterior['time_shift']))
            ext_labels = [r'$\psi  [{\rm rad}]$', r'$t_0  [{\rm s}]$']
        make_corner_plot(ext_matrix,ext_labels,ppdir+'/external_posterior.png')

    except Exception:
        pass

def make_histograms(posterior_samples, names, outdir):
    
    from bajes.inf.utils import autocorrelation
    
    names = np.append(names, ['logL', 'logPrior'])
    
    for i,ni in enumerate(names):
        
        logger.info("... producing histogram for {} ...".format(ni))
    
        try:
            mean    = np.median(posterior_samples[ni])
            upper   = np.percentile(posterior_samples[ni],95)
            lower   = np.percentile(posterior_samples[ni],5)
            
            fig = plt.figure()
            plt.hist(posterior_samples[ni], bins=66, edgecolor = 'royalblue', histtype='step', density=True)
            plt.axvline(mean,   color='navy',ls='--')
            plt.axvline(upper,  color='slateblue',ls='--')
            plt.axvline(lower,  color='slateblue',ls='--')
            plt.ylabel('posterior')
            plt.xlabel('{}'.format(names[i]),size=12)
            plt.savefig(outdir+'/hist_{}.png'.format(names[i]), dpi=200)
            
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

            plt.savefig(outdir+'/samples_{}.png'.format(names[i]), dpi=200)
            plt.close()
        
        except Exception:
            logger.warning("Unable to produce histogram plot for {}, an exception occurred.".format(ni))
            pass


def reconstruct_waveform(outdir, posterior, container_inf, container_gw, whiten=False, N_samples=0, M_tot=None):

    nsub_panels  = len(container_gw.datas.keys())
    strains_dets = {det: {} for det in container_gw.datas.keys()}
    wfs          = {det: [] for det in container_gw.datas.keys()}

    first_det                                 = list(container_gw.datas.keys())[0]
    data_first_det                            = container_gw.datas[first_det]
    freqs, f_min, f_max, t_gps, seglen, srate = data_first_det.freqs, data_first_det.f_min, data_first_det.f_max, data_first_det.t_gps, data_first_det.seglen, data_first_det.srate

    posterior = posterior
    names     = container_inf.prior.names
    constants = container_inf.prior.const
    approx    = container_inf.like.wave.approx
    w         = Waveform(freqs=freqs, srate=srate, seglen=seglen, approx=approx)

    for det in strains_dets.keys():

        strains_dets[det]['s'] = container_gw.datas[det]
        strains_dets[det]['d'] = container_gw.dets[det]
        strains_dets[det]['n'] = container_gw.noises[det]

        strains_dets[det]['d'].store_measurement(strains_dets[det]['s'], strains_dets[det]['n'])

        if(whiten):
            if not(M_tot==None):
                #Estimate of the ringdown frequency, approximating M_final with M_tot and using Schwarzschild value
                f_ringdown = ((CLIGHT_SI**3)/(2.*np.pi*GNEWTON_SI*M_tot*MSUN_SI)) * (1.5251-1.1568*(1-0.7)**0.1292)
                f_max_bp = 2*f_ringdown
            else:
                # Avoid being close to Nyquist frequency when bandpassing.
                f_max_bp = f_max-10

            strains_dets[det]['s'].bandpassing(flow=f_min, fhigh=f_max_bp)
            strains_dets[det]['s'].whitening(strains_dets[det]['n'])

    logger.info("Plotting the reconstructed waveform.")

    if ((N_samples==0) or (N_samples > len(posterior))): samples_list = np.arange(0,len(posterior))
    else:                                                samples_list = np.random.choice(len(posterior), N_samples, replace=False)

    for j,k in enumerate(samples_list):

        # Every 100 steps, update the user on the status of the plot.
        if(j%100==0): logger.info("Progress: {}/{}".format(j+1, len(posterior)))

        params = {name: posterior[name][k] for name in names}
        p      = {**params,**constants}
        hphc   = w.compute_hphc(p)
        h      = (hphc.plus +1j*hphc.cross)
        hphc   = PolarizationTuple(plus=np.real(h), cross=np.imag(h))

        for det in strains_dets.keys():

            h_tmp = strains_dets[det]['d'].project_tdwave(hphc, p, w.domain)
            h_strain = Series('time', h_tmp, seglen=seglen, srate=srate, t_gps=t_gps, f_min=f_min, f_max=f_max)
            if(whiten):
                h_strain.whitening(strains_dets[det]['n'])
                wfs[det].append(h_strain.time_series/np.sqrt(srate))
            else:
                wfs[det].append(h_strain.time_series)

    # Plot the data
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(hspace = .001)

    # plot median, upper, lower and save
    for i,det in enumerate(strains_dets.keys()):

        lo, me, hi = np.percentile(wfs[det],[5,50,95], axis=0)

        ax = fig.add_subplot(nsub_panels,1,i+1)

        if(whiten):
            ax.plot(strains_dets[det]['s'].times-t_gps, strains_dets[det]['s'].time_series/np.sqrt(srate), c='black', linestyle='--', lw=1.0, label='Data')
        else:
            ax.plot(strains_dets[det]['s'].times-t_gps, strains_dets[det]['s'].time_series, c='black', linestyle='--', lw=1.0, label='Data')
        ax.plot(strains_dets[det]['s'].times-t_gps, me, c='gold', lw=0.8, label='Waveform')
        ax.fill_between(strains_dets[det]['s'].times-t_gps, lo, hi, color='gold', alpha=0.4, lw=0.5)

        ax.set_xlabel('t - t$_{\mathrm{gps}}$')
        ax.set_ylabel('Strain {}'.format(det))
        ax.legend(loc='upper right', prop={'size': 6})
        if not(i==len(strains_dets.keys())-1):
            ax.get_xaxis().set_visible(False)

        wf_ci_fl = open(outdir +'/wf_ci_{}.dat'.format(det),'w')
        wf_ci_fl.write('#\t t \t median \t lower \t higher\n')
        for i in range(len(strains_dets[det]['s'].times)):
            wf_ci_fl.write("%.10f \t %.10f \t %.10f \t %.10f \n" %(strains_dets[det]['s'].times[i], me[i], lo[i], hi[i]))
        wf_ci_fl.close()
        
    if(whiten): plt.savefig(outdir +'/Reconstructed_waveform_whitened.pdf', bbox_inches='tight')
    else:       plt.savefig(outdir +'/Reconstructed_waveform.pdf', bbox_inches='tight')

    if not(M_tot==None):    
        # Repeating the above plot while zooming on the merger.
        # FIXME: this could be done in a single shot, storing the axes.
        fig = plt.figure(figsize=(8,6))
        plt.subplots_adjust(hspace = .001)

        for i,det in enumerate(strains_dets.keys()):

            lo, me, hi = np.percentile(wfs[det],[5,50,95], axis=0)

            ax = fig.add_subplot(nsub_panels,1,i+1)

            # Compute the t_peak of the median waveform, relative to the gps time
            t_peak = strains_dets[det]['s'].times[np.argmax(me)]-t_gps

            if(whiten):
                ax.plot(strains_dets[det]['s'].times-t_gps, strains_dets[det]['s'].time_series/np.sqrt(srate), c='black', linestyle='--', lw=1.0, label='Data')
            else:
                ax.plot(strains_dets[det]['s'].times-t_gps, strains_dets[det]['s'].time_series, c='black', linestyle='--', lw=1.0, label='Data')
            ax.plot(strains_dets[det]['s'].times-t_gps, me, c='gold', lw=0.8, label='Waveform')
            ax.fill_between(strains_dets[det]['s'].times-t_gps, lo, hi, color='gold', alpha=0.4, lw=0.5)

            ax.set_xlim([t_peak-200*M_tot*MTSUN_SI, t_peak+200*M_tot*MTSUN_SI])
            ax.set_xlabel('t - t$_{\mathrm{gps}}$')
            ax.set_ylabel('Strain {}'.format(det))
            ax.legend(loc='upper right', prop={'size': 6})
            if not(i==len(strains_dets.keys())-1):
                ax.get_xaxis().set_visible(False)

            wf_ci_fl = open(outdir +'/wf_ci_{}.dat'.format(det),'w')
            wf_ci_fl.write('#\t t \t median \t lower \t higher\n')
            for i in range(len(strains_dets[det]['s'].times)):
                wf_ci_fl.write("%.10f \t %.10f \t %.10f \t %.10f \n" %(strains_dets[det]['s'].times[i], me[i], lo[i], hi[i]))
            wf_ci_fl.close()
            
        if(whiten): plt.savefig(outdir +'/Reconstructed_waveform_whitened_zoom.pdf', bbox_inches='tight')
        else:       plt.savefig(outdir +'/Reconstructed_waveform_zoom.pdf', bbox_inches='tight')



if __name__ == "__main__":

    parser=op.OptionParser()
    parser.add_option('-p','--post',      dest='posterior',       default=None,           type='string',                        help="Posterior file to postprocess.")
    parser.add_option('-o','--outdir',    dest='outdir',          default=None,           type='string',                        help="Name of the output directory.")
    
    parser.add_option('--M-tot-estimate', dest='M_tot',           default=None,                                                 help="Optional: Estimate of the total mass of the system, if not None, it is used to set narrower bandpassing and merger zoom. If equal to 'posterior', the value is extracted from the posterior samples. If a float is passed, that value is used instead. Default: None.")
    parser.add_option('--N-samples-wf',   dest='N_samples_wf',    default=1000,           type='int',                           help="Optional: Number of samples to be used in waveform reconstruction. If 0, all samples are used. Default: 1000.")
    parser.add_option('--spin-flag',      dest='spin_flag',       default='no-spins',     type='string',                        help="Optional: Spin prior flag. Default: 'no-spins'. Available options: ['no-spins', 'align', 'precess'].")
    parser.add_option('--tidal-flag',     dest='lambda_flag',     default='no-tides',     type='string',                        help="Optional: Spin prior flag. Default: 'no-tides'. Available options: ['no-tides', 'bns-tides', 'bhns-tides'].")
    parser.add_option('--engine',         dest='engine',          default='Unknown',      type='string',                        help="Optional: Engine sampler label. Default: 'Unknown'. Currently UNUSED.")
    parser.add_option('--extra-flag',     dest='extra_flag',      default='',             type='string',      action="append",  help="Optional: Extra prior flag. Default: ''. Currently UNUSED.")
    (opts,args) = parser.parse_args()

    if not(opts.outdir==None): outdir = opts.outdir
    else: raise ValueError("The 'outdir' option is a mandatory parameter. Aborting.")
    
    ppdir  = os.path.abspath(outdir+'/postproc')
    wf_dir = os.path.abspath(outdir+'/postproc/Waveform_reconstruction') 
    ensure_dir(ppdir)
    ensure_dir(wf_dir)

    global logger

    logger = set_logger(outdir=ppdir, label='bajes_postproc')
    logger.info("Running bajes postprocessing:")
    logger.info("The reported uncertainties correpond to 90% credible regions.")
    logger.info("The contours of the corner plots represent 50%, 90% credible regions.")

    if not(opts.posterior==None): posterior = np.genfromtxt(opts.posterior, names=True)
    else: raise ValueError("The 'post' option is a mandatory parameter. Aborting.")
    
    # extract prior object from pickle
    if not(os.path.exists(outdir + '/run')): dc    = data_container(outdir + '/inf.pkl')
    else:                                    dc    = data_container(outdir + 'run/inf.pkl')
    if not(os.path.exists(outdir + '/run')): dc_gw = data_container(outdir + '/gw_obs.pkl')
    else:                                    dc_gw = data_container(outdir + 'run/gw_obs.pkl')
    container_inf = dc.load()
    container_gw  = dc_gw.load()
    priors        = container_inf.prior

    # produce histogram plots
    logger.info("Producing histograms...")
    ensure_dir(ppdir+'/histgr')
    make_histograms(posterior, priors.names, ppdir+'/histgr')

    # produce corner plots
    logger.info("Producing corners...")
    ensure_dir(ppdir+'/corner')
    make_corners(posterior, opts.spin_flag, opts.lambda_flag, opts.extra_flag, ppdir+'/corner')

    if not(opts.M_tot==None):
        if(opts.M_tot=='posterior'):
            if('mtot' in prior.names):
                opts.M_tot = np.median(posterior['mtot'])
            elif('mtot' in prior.const):
                opts.M_tot = prior.const['mtot']
            elif(('mc' in prior.names) and ('q' in prior.names)):
                opts.M_tot = mcq_to_m1(np.median(posterior['mc']), np.median(posterior['q'])) + mcq_to_m2(np.median(posterior['mc']), np.median(posterior['q']))
            elif(('mc' in prior.names) and ('q' in prior.const)):
                opts.M_tot = mcq_to_m1(np.median(posterior['mc']), prior.const['q']) + mcq_to_m2(np.median(posterior['mc']), prior.const['q'])
            elif(('mc' in prior.const) and ('q' in prior.names)):
                opts.M_tot = mcq_to_m1(prior.const['mc'], np.median(posterior['q'])) + mcq_to_m2(prior.const['mc'], np.median(posterior['q']))
            elif(('mc' in prior.const) and ('q' in prior.const)):
                opts.M_tot = mcq_to_m1(prior.const['mc'], prior.const['q']) + mcq_to_m2(prior.const['mc'], prior.const['q'])
            else:
                logger.warning("Could not extract M_tot (either directly or through related mass parameters) from posterior or fixed parameters. Setting it to None and skipping zoomed plots.")
                opts.M_tot = None
        else:
            opts.M_tot = float(opts.M_tot)

    # produce waveform plots
    logger.info("Reconstructing waveforms...")
    reconstruct_waveform(wf_dir, posterior, container_inf, container_gw, whiten=False, N_samples = opts.N_samples_wf, M_tot = opts.M_tot)
    logger.info("Reconstructing whitened waveforms...")
    reconstruct_waveform(wf_dir, posterior, container_inf, container_gw, whiten=True,  N_samples = opts.N_samples_wf, M_tot = opts.M_tot)

    logger.info("... done.")
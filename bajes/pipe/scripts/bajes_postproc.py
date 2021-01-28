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

from bajes.pipe import ensure_dir, data_container, cart2sph, sph2cart, set_logger
from bajes.obs.gw.utils import compute_tidal_components, compute_lambda_tilde, compute_delta_lambda

def make_corner_plot(matrix , labels, outputname):

    N = len(labels)
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
    except KeyError:
        pass

    # spins
    if 'align' in spin_flag:
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

    elif 'precess' in spin_flag:

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

    # tides
    if lambda_flag == 'bns-tides':
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

    elif lambda_flag == 'bhns-tides' or lambda_flag == 'nsbh-tides' :

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

if __name__ == "__main__":

    parser=op.OptionParser()
    parser.add_option('-p','--post',    default=None,           type='string',          dest='posterior',   help='posterior file')
    parser.add_option('-o','--outdir',  default=None,           type='string',          dest='outdir',      help='directory for output')
    
    parser.add_option('--engine',       dest='engine',          default='Unknown',      type='string',      help='engine sampler label')
    parser.add_option('--spin-flag',    dest='spin_flag',       default='no-spins',     type='string',      help='spin prior flag')
    parser.add_option('--tidal-flag',   dest='lambda_flag',     default='no-tides',     type='string',      help='tidal prior flag')
    parser.add_option('--extra-flag',   dest='extra_flag',      default='',             type='string',      action="append",  help='extra prior flag')
    (opts,args) = parser.parse_args()

    ppdir = os.path.abspath(opts.outdir+'/postproc')
    ensure_dir(ppdir)
    
    global logger
    
    logger = set_logger(outdir=ppdir, label='bajes_postproc')
    logger.info("Running bajes postprocessing:")
    logger.info("The reported uncertainties correpond to 90% credible regions.")
    logger.info("The contours of the corner plots represent 50%, 90% credible regions.")

    # extract posterior samples coming from sampler
    posterior = np.genfromtxt(opts.posterior , names=True)
    
    # extract prior object from pickle
    dc          = data_container(opts.outdir + '/run/inf.pkl')
    container   = dc.load()
    priors      = container.prior
    
    # produce histogram plots
    ensure_dir(ppdir+'/histgr')
    make_histograms(posterior, priors.names, ppdir+'/histgr')

    # produce corner plots
    ensure_dir(ppdir+'/corner')
    make_corners(posterior, opts.spin_flag, opts.lambda_flag, opts.extra_flag, ppdir+'/corner')

    logger.info("... done.")


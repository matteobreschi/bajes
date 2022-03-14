#!/usr/bin/env python
from __future__ import division, unicode_literals
import sys
import os
import logging
import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from bajes.pipe import ensure_dir, execute_bash, set_logger

class BajesPipeError(Exception):
    pass

def write_executable(outdir, config, string1, string2, string3):

    # checking if executable is already in outdir
    # if yes, the process is aborted
    execname = '{}.sub'.format(config['pipe']['jobname'])
    listdir = os.listdir(outdir)
    if execname in listdir:
        logger.error("An executable file {} is already in outdir={}. Please select another output directory or remove the executable from this folder.".format(execname , outdir))
        raise BajesPipeError("An executable file {} is already in outdir={}. Please select another output directory or remove the executable from this folder.".format(execname , outdir))

    execname = outdir+'/'+execname
    execfile = open(execname, 'w')

    if config['pipe']['sub'] == 'bash':
        execfile.write('#!/bin/bash'+'\n')
        execfile.write('\n')

        if string1:
            execfile.write(string1+'\n')
            execfile.write('\n')
        if string2:
            execfile.write(string2+'\n')
            execfile.write('\n')
        if string3:
            execfile.write(string3+'\n')
            execfile.write('\n')

    elif config['pipe']['sub'] == 'slurm':

        list_keys_in_pipe = np.transpose(list(config.items('pipe')))[0]

        # setting string for main commands
        srun_string = 'time srun '

        # set SLURM variables
        nnodes  = 1
        ntasks  = 1
        ncpu    = int(config['pipe']['nprocs'])

        # setting string for core command
        srun_string_core = srun_string

        # set mpi flag if it is missing
        if 'mpi' not in list_keys_in_pipe:
            config['pipe']['mpi'] = 0

        # set n-nodes if mpi is on
        if int(config['pipe']['mpi']):

            try:
                nnodes = config['pipe']['nnodes']
            except KeyError:
                logger.warning("Requested MPI parallelization but number of nodes is missing. Setting number of nodes equal to 1.")
                nnodes = 1

        # set n-tasks
        if 'cpu-per-task' in list_keys_in_pipe:
            cpu_per_task = int(config['pipe']['cpu-per-task'])
            ntasks       = ncpu/cpu_per_task
        else:
            if int(config['pipe']['mpi']):
                logger.warning("Using MPI parallelization without setting cpu-per-task. The default is cpu-per-task = 1")
                ntasks = ncpu
                cpu_per_task = 1
            else:
                cpu_per_task = ncpu

        # check
        if not int(config['pipe']['mpi']) and ntasks>1:
            logger.error("MPI parallelization is not requested but number of tasks is greater than 1. Please check your settings.")
            raise BajesPipeError("MPI parallelization is not requested but number of tasks is greater than 1. Please check your settings.")

        # set MPI
        mpi_per_node = 1

        if int(config['pipe']['mpi']):

            # set mpi_per_node
            mpi_per_node =  int(int(ntasks)/int(nnodes))

            # get mpi
            if 'mpi-type' in list_keys_in_pipe:
                which_mpi = config['pipe']['mpi-type']
            else:
                logger.warning("Process management interface not specified, using PMI-2.")
                which_mpi = 'pmi2'

            # set up core string
            srun_string_core    += '-n $SLURM_NTASKS --mpi={} '.format(which_mpi)

        # writing slurm configuation
        execfile.write('#!/bin/bash'+'\n')
        execfile.write('#SBATCH --job-name={}'.format(config['pipe']['jobname'])+'\n')

        if 'partition' in list_keys_in_pipe:
            execfile.write('#SBATCH --partition {}'.format(config['pipe']['partition'])+'\n')
        else:
            logger.error("Unable to write pipeline for SLURM. Please include partition information in config.")
            raise BajesPipeError("Unable to write pipeline for SLURM. Please include partition information in config.")

        if 'mail' in list_keys_in_pipe:
            execfile.write('#SBATCH --mail-type=ALL'+'\n')
            execfile.write('#SBATCH --mail-user={}'.format(config['pipe']['mail'])+'\n')

        execfile.write('#SBATCH --time={}'.format(config['pipe']['walltime'])+'\n')
        execfile.write('#SBATCH --nodes={}'.format(nnodes)+'\n')
        execfile.write('#SBATCH --ntasks-per-node={}'.format(mpi_per_node)+'\n')
        execfile.write('#SBATCH --cpus-per-task={}'.format(cpu_per_task)+'\n')

        if 'mem-per-cpu' in list_keys_in_pipe:
            execfile.write('#SBATCH --mem-per-cpu={}'.format(config['pipe']['mem-per-cpu'])+'\n')
        else:
            logger.warning("Memory-per-CPU not specified, using 1G.")
            execfile.write('#SBATCH --mem-per-cpu=1G '+'\n')

        execfile.write('#SBATCH -o {}/bajes.out'.format(config['pipe']['outdir'])+'\n')
        execfile.write('#SBATCH -e {}/bajes.err'.format(config['pipe']['outdir'])+'\n')
        execfile.write('\n')

        # writing modules to be load
        if 'module' in list_keys_in_pipe:

            if ',' in config['pipe']['module']:

                paths   = config['pipe']['module'].split(',')
                for pi in paths:
                    execfile.write('module load {}'.format(pi) + '\n')

            else:
                execfile.write('module load {}'.format(config['pipe']['module']) + '\n')

            execfile.write('\n')

        # writing paths to be sourced
        if 'source' in list_keys_in_pipe:

            if ',' in config['pipe']['source']:

                paths   = config['pipe']['source'].split(',')
                for pi in paths:
                    execfile.write('source {}'.format(pi) + '\n')

            else:
                execfile.write('source {}'.format(config['pipe']['source']) + '\n')

            execfile.write('\n')

        # writing variables to be exported
        if 'export' in list_keys_in_pipe:

            if ',' in config['pipe']['export']:

                paths   = config['pipe']['export'].split(',')
                for pi in paths:
                    execfile.write('export {}'.format(pi) + '\n')

            else:
                execfile.write('export {}'.format(config['pipe']['export']) + '\n')

            execfile.write('\n')

        # export OMP_NUM_THREADS=1
        # turns off the OpenMP multi-threading,
        # so each Python process remains single-threaded
        execfile.write('export OMP_NUM_THREADS=1'+'\n')
        # execfile.write('export OMP_DYNAMIC=0'+'\n')

        # same for MKL
        execfile.write('export MKL_NUM_THREADS=1'+'\n')
        execfile.write('export MKL_DYNAMIC=0'+'\n')

        if int(config['pipe']['mpi']):
            execfile.write('export MPI_PER_NODE={}'.format(mpi_per_node)+'\n')
            string2 += ' --mpi-per-node {}'.format(mpi_per_node)

        # writing path to be exported for bajes
        execfile.write('\n')

        # writing job info
        execfile.write('echo "Date              = $(date)"' + '\n')
        execfile.write('echo "Hostname          = $(hostname -s)"' + '\n')
        execfile.write('echo "Working Directory = $(pwd)"' + '\n')
        execfile.write('\n')
        execfile.write('echo "Names  of Nodes     = $SLURM_NODELIST"' + '\n')
        execfile.write('echo "Number of Nodes     = $SLURM_JOB_NUM_NODES"' + '\n')
        execfile.write('echo "Number of Tasks     = $SLURM_NTASKS"' + '\n')
        execfile.write('echo "Number of CPU/Task  = $SLURM_CPUS_PER_TASK"' + '\n')
        execfile.write('\n')

        # writing main commands
        if string1:
            execfile.write(srun_string  + string1 + '\n')
            execfile.write('\n')
        if string2:
            execfile.write(srun_string_core + string2 + '\n')
            execfile.write('\n')
        if string3:
            execfile.write(srun_string  + string3 + '\n')
            execfile.write('\n')

    execfile.close()

    # create executable
    bashcommand = 'chmod u+x {}'.format(execname)
    execute_bash(bashcommand)
    return execname

def write_inject_string(config, ifos, outdir):
    """
        Write command string to execute bajes_inject.py
        given a config file
    """

    injection_string = 'bajes_inject.py --outdir {} '.format(outdir+'/injection/')

    try:
        injpath = os.path.abspath(config['gw-data']['inj-strain'])
        injection_string += '--wave {} '.format(injpath)
    except KeyError:
        logger.error("Invalid or missing inj-strain in config file. The file must contains the time-domain waveform (sampled at given srate) to be injected, with the following columns: time, reh, imh")
        raise BajesPipeError("Invalid or missing inj-strain in config file. The file must contains the time-domain waveform (sampled at given srate) to be injected, with the following columns: time, reh, imh")

    for ifo in ifos:
        try:
            injection_string += '--ifo {} --asd {} '.format(ifo , config['gw-data']['{}-asd'.format(ifo)])
        except KeyError:
            logger.error("Invalid or missing {}-asd in config file. Please include ASD files for every IFO.".format(ifo))
            raise BajesPipeError("Invalid or missing {}-asd in config file. Please include ASD files for every IFO.".format(ifo))

    injection_string += '--seglen {} '.format(config['gw-data']['seglen'])
    injection_string += '--srate {} '.format(config['gw-data']['srate'])

    try:
        if config['gw-data']['window']:
            injection_string += '--window {} '.format(config['gw-data']['window'])
    except Exception:
        pass

    try:
        if config['gw-data']['alpha']:
            injection_string += '--tukey {} '.format(config['gw-data']['alpha'])
    except Exception:
        pass

    try:
        if (int (config['gw-data']['zero-noise'])) :
            injection_string += '--zero-noise '
    except KeyError:
        pass

    injection_string += '--f-min {} '.format(config['gw-data']['f-min'])
    injection_string += '--t-gps {} '.format(config['gw-data']['t-gps'])

    try:
        injection_string += '--ra {} '.format(config['gw-data']['inj-ra'])
    except Exception:
        pass

    try:
        injection_string += '--dec {} '.format(config['gw-data']['inj-dec'])
    except Exception:
        pass

    try:
        injection_string += '--pol {} '.format(config['gw-data']['inj-pol'])
    except Exception:
        pass

    return injection_string

def write_gwosc_string(config, ifos, outdir):
    """
        Write command string to execute bajes_read_gwosc.py
        given a config file
    """

    read_string = 'bajes_read_gwosc.py --outdir {} '.format(outdir)

    try:
        read_string += '--event {} '.format(config['gw-data']['event'])
    except Exception:
        pass

    try:
        read_string += '--version {} '.format(config['gw-data']['version'])
    except Exception:
        pass

    for ifo in ifos:
        read_string += '--ifo {} '.format(ifo)

    read_string += '--seglen {} '.format(config['gw-data']['seglen'])
    read_string += '--srate {} '.format(config['gw-data']['srate'])
    read_string += '--t-gps {} '.format(config['gw-data']['t-gps'])

    return read_string


def write_run_string(config, tags, outdir):
    """
        Write command string to execute bajes_core.py
        given a config file
    """

    list_keys_in_pipe       = np.transpose(list(config.items('pipe')))[0]
    list_keys_in_sampler    = np.transpose(list(config.items('sampler')))[0]

    run_string = 'bajes_core.py '
    if 'mpi' in list_keys_in_pipe:
        if int(config['pipe']['mpi']):
            run_string = 'bajes_parallel_core.py '

    run_string += '--outdir {} '.format(outdir+'/run/')

    if config['sampler']['engine'] == 'cpnest':

        run_string += '--engine {} '.format(config['sampler']['engine'])
        run_string += '--nlive {} '.format(config['sampler']['nlive'])

        if 'maxmcmc' in list_keys_in_sampler:
            run_string += '--maxmcmc {} '.format(config['sampler']['maxmcmc'])

        if 'tolerance' in list_keys_in_sampler:
            run_string += '--tol {} '.format(config['sampler']['tolerance'])

        if 'poolsize' in list_keys_in_sampler:
            run_string += '--poolsize {} '.format(config['sampler']['poolsize'])

    elif config['sampler']['engine'] == 'ultranest':

        run_string += '--engine {} '.format(config['sampler']['engine'])
        run_string += '--nlive {} '.format(config['sampler']['nlive'])

        if 'tolerance' in list_keys_in_sampler:
            run_string += '--tol {} '.format(config['sampler']['tolerance'])

        if 'maxmcmc' in list_keys_in_sampler:
            run_string += '--maxmcmc {} '.format(config['sampler']['maxmcmc'])

        if 'minmcmc' in list_keys_in_sampler:
            run_string += '--minmcmc {} '.format(config['sampler']['minmcmc'])

        if 'nout' in list_keys_in_sampler:
            run_string += '--nout {} '.format(config['sampler']['nout'])

        if 'dkl' in list_keys_in_sampler:
            run_string += '--dkl {} '.format(config['sampler']['dkl'])

        if 'z-frac' in list_keys_in_sampler:
            run_string += '--z-frac {} '.format(config['sampler']['z-frac'])

    elif config['sampler']['engine'] == 'dynesty' or config['sampler']['engine'] == 'dynesty-dyn':

        run_string += '--engine {} '.format(config['sampler']['engine'])
        run_string += '--nlive {} '.format(config['sampler']['nlive'])

        if 'nbatch' in list_keys_in_sampler:
            run_string += '--nbatch {} '.format(config['sampler']['nbatch'])

        if 'maxmcmc' in list_keys_in_sampler:
            run_string += '--maxmcmc {} '.format(config['sampler']['maxmcmc'])

        if 'minmcmc' in list_keys_in_sampler:
            run_string += '--minmcmc {} '.format(config['sampler']['minmcmc'])

        if 'tolerance' in list_keys_in_sampler:
            run_string += '--tol {} '.format(config['sampler']['tolerance'])

        if 'nact' in list_keys_in_sampler:
            run_string += '--nact {} '.format(config['sampler']['nact'])

    elif config['sampler']['engine'] == 'emcee' or config['sampler']['engine'] == 'ptmcmc':

        run_string += '--engine {} '.format(config['sampler']['engine'])
        run_string += '--nwalk {} '.format(config['sampler']['nwalk'])
        run_string += '--nout {} '.format(config['sampler']['nout'])

        if config['sampler']['engine'] == 'ptmcmc':
            if 'ntemp' not in list_keys_in_sampler:
                logger.error("Unable to read number of parallel temperature. Please specify the number of temperatures if you want to use ptmcmc.")
                raise BajesPipeError("Unable to read number of parallel temperature. Please specify the number of temperatures if you want to use ptmcmc.")
            else:
                run_string += '--ntemp {} '.format(config['sampler']['ntemp'])

        if 'nburn' in list_keys_in_sampler:
            run_string += '--nburn {} '.format(config['sampler']['nburn'])

        if 'tmax' in list_keys_in_sampler:
            run_string += '--tmax {} '.format(config['sampler']['tmax'])

    else:
        from bajes.inf import __known_samplers__
        logger.error("Invalid string for sampler engine. Plese use one of the following: {}".format(__known_samplers__))
        raise BajesPipeError("Invalid string for engine.")

    run_string += '--nprocs {} '.format(config['pipe']['nprocs'])

    if 'mpi' in list_keys_in_pipe:
        if 'mpi-fast' in list_keys_in_pipe:
            if int(config['pipe']['mpi-fast']):
                run_string += '--fast-mpi '

    if 'seed' in list_keys_in_sampler:
        run_string += '--seed {} '.format(config['sampler']['seed'])

    if 'ncheck' in list_keys_in_sampler:
        run_string += '--checkpoint {} '.format(config['sampler']['ncheck'])

    if 'slice' in list_keys_in_sampler:
        if 'dynesty' in config['sampler']['engine'] or config['sampler']['engine'] == 'ptmcmc':
            logger.warning("Unable to use slice proposal with requested sampler. Option not implemented or already existing.")
        else:
            if int(config['sampler']['slice']):
                run_string += '--use-slice '

    if 'gw' in tags:

        run_string += '--tag gw '

        ifos = config['gw-data']['ifos'].split(',')
        list_keys_in_data       = np.transpose(list(config.items('gw-data')))[0]
        list_keys_in_prior      = np.transpose(list(config.items('gw-prior')))[0]

        if config['gw-data']['data-flag'] == 'local':
            for ifo in ifos:
                run_string += '--ifo {} --asd {} --strain {} '.format(ifo , config['gw-data']['{}-asd'.format(ifo)],
                                                                    config['gw-data']['{}-strain'.format(ifo)])

        elif config['gw-data']['data-flag'] == 'inject':
            for ifo in ifos:
                run_string += '--ifo {} --asd {} --strain {} '.format(ifo , config['gw-data']['{}-asd'.format(ifo)],
                                                                   outdir+'/injection/{}_INJECTION.txt'.format(ifo))

        elif config['gw-data']['data-flag'] == 'gwosc':
            for ifo in ifos:
                run_string += '--ifo {} --asd {} --strain {} '.format(ifo , config['gw-data']['{}-asd'.format(ifo)],
                                                                    outdir+'/data/{}_STRAIN_{}_{}_{}.txt'.format(ifo ,
                                                                                                                 config['gw-data']['seglen'],
                                                                                                                 config['gw-data']['srate'],
                                                                                                                 config['gw-data']['t-gps']))

        run_string += '--seglen {} '.format(config['gw-data']['seglen'])
        run_string += '--srate {} '.format(config['gw-data']['srate'])
        run_string += '--f-min {} '.format(config['gw-data']['f-min'])
        run_string += '--f-max {} '.format(config['gw-data']['f-max'])
        run_string += '--t-gps {} '.format(config['gw-data']['t-gps'])
        run_string += '--data-flag {} '.format(config['gw-data']['data-flag'])

        if 'alpha' in list_keys_in_data:
            run_string += '--alpha {}'.format(config['gw-data']['alpha'])

        run_string += '--approx {} '.format(config['gw-prior']['approx'])
        run_string += '--spin-flag {} '.format(config['gw-prior']['spin-flag'])
        run_string += '--tidal-flag {} '.format(config['gw-prior']['tidal-flag'])
        run_string += '--mc-min {} '.format(config['gw-prior']['mchirp-min'])
        run_string += '--mc-max {} '.format(config['gw-prior']['mchirp-max'])
        run_string += '--q-max {} '.format(config['gw-prior']['q-max'])

        if 'm-max' in list_keys_in_prior:
            run_string += '--mass-max {} '.format(config['gw-prior']['m-max'])
        if 'm-min' in list_keys_in_prior:
            run_string += '--mass-min {} '.format(config['gw-prior']['m-min'])

        run_string += '--dist-min {} '.format(config['gw-prior']['dist-min'])
        run_string += '--dist-max {} '.format(config['gw-prior']['dist-max'])

        if 'tshift-max' in list_keys_in_prior:
            run_string += '--tshift-max {} '.format(config['gw-prior']['tshift-max'])
        else:
            logger.warning("Using default upper bound for time shift (1s). If the signal is not centered on the given data segment, this may create problems.")
            run_string += '--tshift-max {} '.format(1)

        if 'tshift-min' in list_keys_in_prior:
            run_string += '--tshift-min {} '.format(config['gw-prior']['tshift-min'])

        if 'spin-max' in list_keys_in_prior:
            run_string += '--spin-max {} '.format(config['gw-prior']['spin-max'])

        if 'lambda-min' in list_keys_in_prior:
            run_string += '--lambda-min {} '.format(config['gw-prior']['lambda-min'])

        if 'lambda-max' in list_keys_in_prior:
            run_string += '--lambda-max {} '.format(config['gw-prior']['lambda-max'])

        if 'dist-flag' in list_keys_in_prior:
            run_string += '--dist-flag {} '.format(config['gw-prior']['dist-flag'])

        if 'prior-grid' in list_keys_in_prior:
            run_string += '--priorgrid {} '.format(config['gw-prior']['prior-grid'])

        if 'marg-phi' in list_keys_in_prior:
            if (int (config['gw-prior']['marg-phi'])) :
                run_string += '--marg-phi-ref '

        if 'l-max' in list_keys_in_prior:
            lmax = int(config['gw-prior']['l-max'])
            if lmax == 0 :
                logger.warning("Requested lmax is 0. The waveform will include only the 22-mode.")
            elif lmax == 1 :
                logger.warning("Requested lmax is 1, this is not allowed for GW signals. Using only the 22-mode.")
            else:
                run_string += '--lmax {} '.format(lmax)

        if 'ej-flag' in list_keys_in_prior:

            if int(config['gw-prior']['ej-flag']):

                run_string += '--use-energy-angmom '

                if 'en-min' in list_keys_in_prior and 'en-max' in list_keys_in_prior:
                    run_string += '--e-min {} --e-max {} '.format(config['gw-prior']['en-min'],config['gw-prior']['en-max'])
                else:
                    logger.warning("Impossible to read bounds for energy parameter. Using default option.")

                if 'j-min' in list_keys_in_prior and 'j-max' in list_keys_in_prior:
                    run_string += '--j-min {} --j-max {} '.format(config['gw-prior']['j-min'],config['gw-prior']['j-max'])
                else:
                    logger.warning("Impossible to read bounds for angular momentum parameter. Using default option.")

        if 'ecc-flag' in list_keys_in_prior:

            if int(config['gw-prior']['ecc-flag']):

                run_string += '--use-eccentricity '

                if 'ecc-min' in list_keys_in_prior and 'ecc-max' in list_keys_in_prior:
                    run_string += '--ecc-min {} --ecc-max {} '.format(config['gw-prior']['ecc-min'],config['gw-prior']['ecc-max'])
                else:
                    logger.warning("Impossible to read bounds for eccentricity parameter. Using default option.")

        using_binning = 0

        if 'binning' in list_keys_in_sampler:
            if (int (config['sampler']['binning'])) :
                run_string += '--use-binning '
                using_binning = 1

                if 'fiducial' in list_keys_in_data:
                    run_string += '--fiducial {} '.format(config['gw-data']['fiducial'])
                else:
                    logger.warning("Impossible to read fiducial waveform for GWBinning. Default option assumes that you have the params.ini file in your outdir.")

        if 'marg-time' in list_keys_in_prior:
            if (int (config['gw-prior']['marg-time'])) :
                if 'kn' in tags:
                    logger.warning("Requested time shift marginalization for GW likelihood, but the KN likelihood marginalization is unknown. Ignorings marg-time flag.")
                else:
                    if using_binning:
                        logger.warning("Time shift marginalization not available with frequency binning")
                    else:
                        run_string += '--marg-time-shift '

        if 'psd-weights' in list_keys_in_prior:
            if int(config['gw-prior']['psd-weights']) :
                if using_binning:
                    logger.warning("PSD weights not available with frequency binning")
                else:
                    run_string += '--psd-weights {} '.format(config['gw-data']['psd-weights'])

        for i,ifo in enumerate(ifos):
            if '{}-calib'.format(ifo) in list_keys_in_data:
                run_string += '--spcal {} '.format(config['gw-data']['{}-calib'.format(ifo)])
            else:
                logger.warning("Unable to read calibration error file for {} IFO. Calibration errors will be ignored.".format(ifo))
                config['gw-prior']['spcal-nodes'] = '0'

        if 'spcal-nodes' in list_keys_in_prior:
            run_string += '--nspcal {} '.format(config['gw-prior']['spcal-nodes'])

        for ki in list_keys_in_prior:
            if 'fix' in ki:
                fix_name    = ki.split('-')[1]
                fix_value   = config['gw-prior']['fix-{}'.format(fix_name)]
                run_string += '--fix-name {} --fix-value {}  '.format(fix_name, fix_value)

    if 'kn' in tags:

        run_string += '--tag kn '

        list_keys_in_data       = np.transpose(list(config.items('kn-data')))[0]
        list_keys_in_prior      = np.transpose(list(config.items('kn-prior')))[0]

        try:
            bands = config['kn-data']['photo-bands'].split(',')
            logger.info("... using {} photometric bands ({}) ...".format(len(bands), config['kn-data']['photo-bands']))
        except KeyError:
            logger.error("Invalid or missing bands in config file. Please specify the bands (with comma-separated acronymes) in [kn-data] section.")
            raise BajesPipeError("Invalid or missing bands in config file. Please specify the bands (with comma-separated acronymes) in [kn-data] section.")

        try:
            lambdas = config['kn-data']['photo-lambdas'].split(',')
        except KeyError:
            lambdas = []

        if len(lambdas)==0 or len(lambdas)!=len(bands):
            from bajes.obs.kn import __photometric_bands__
            logger.warning("Invalid or missing photometric wavelength in config file. Using default values.")
            lambdas = [__photometric_bands__[bi] for bi in bands]

        for bi,li in zip(bands,lambdas):
            run_string += '--band {} --lambda {}  '.format(bi, li)

        if 'dered' in list_keys_in_data:
            if int(config['kn-data']['dered']):
                run_string += '--use-dereddening '
        else:
            run_string += '--use-dereddening '

        if 'mag-folder' in list_keys_in_data:
            if config['kn-data']['mag-folder'] == 'AT2017gfo':
                from bajes import __path__
                run_string += '--mag-folder {}'.format(__path__[0]+'/pipe/data/kn/filter/AT2017gfo/')
            else:
                run_string += '--mag-folder {}'.format(config['kn-data']['mag-folder'])
        else:
            logger.error("Missing path to magnitude data folder in config file. Please specify the mag-folder in [kn-data] section.")
            raise BajesPipeError("Missing path to magnitude data folder in config file. Please specify the mag-folder in [kn-data] section.")


        comps       = config['kn-prior']['comps'].split(',')
        mej_max     = config['kn-prior']['mej-max'].split(',')
        mej_min     = config['kn-prior']['mej-min'].split(',')
        vel_max     = config['kn-prior']['vel-max'].split(',')
        vel_min     = config['kn-prior']['vel-min'].split(',')
        opac_max    = config['kn-prior']['opac-max'].split(',')
        opac_min    = config['kn-prior']['opac-min'].split(',')

        if len(comps) != len(mej_max):
            logger.error("Number of components ({}) does not match number of upper bounds for ejected mass ({}).".format(len(comps), len(mej_max)))
            raise BajesPipeError("Number of components ({}) does not match number of upper bounds for ejected mass ({}).".format(len(comps), len(mej_max)))
        if len(comps) != len(mej_min):
            logger.error("Number of components ({}) does not match number of lower bounds for ejected mass ({}).".format(len(comps), len(mej_min)))
            raise BajesPipeError("Number of components ({}) does not match number of lower bounds for ejected mass ({}).".format(len(comps), len(mej_min)))
        if len(comps) != len(vel_max):
            logger.error("Number of components ({}) does not match number of upper bounds for velocity ({}).".format(len(comps), len(vel_max)))
            raise BajesPipeError("Number of components ({}) does not match number of upper bounds for velocity ({}).".format(len(comps), len(vel_max)))
        if len(comps) != len(vel_min):
            logger.error("Number of components ({}) does not match number of lower bounds for velocity ({}).".format(len(comps), len(vel_min)))
            raise BajesPipeError("Number of components ({}) does not match number of lower bounds for velocity ({}).".format(len(comps), len(vel_min)))
        if len(comps) != len(opac_max):
            logger.error("Number of components ({}) does not match number of upper bounds for opacity ({}).".format(len(comps), len(opac_max)))
            raise BajesPipeError("Number of components ({}) does not match number of upper bounds for opacity ({}).".format(len(comps), len(opac_max)))
        if len(comps) != len(opac_min):
            logger.error("Number of components ({}) does not match number of lower bounds for opacity ({}).".format(len(comps), len(opac_min)))
            raise BajesPipeError("Number of components ({}) does not match number of lower bounds for opacity ({}).".format(len(comps), len(opac_min)))

        for ci, mui, mli, vui, vli, oui, oli in zip(comps, mej_max, mej_min, vel_max, vel_min, opac_max, opac_min):
            run_string += '--comp {} --mej-max {} --mej-min {} --vel-max {} --vel-min {} --opac-max {} --opac-min {} '.format(ci, mui, mli, vui, vli, oui, oli)

        run_string += '--dist-min {} '.format(config['kn-prior']['dist-min'])
        run_string += '--dist-max {} '.format(config['kn-prior']['dist-max'])

        if 'dist-flag' in list_keys_in_prior:
            # check that the flag is not repeated
            if '--dist-flag' not in run_string:
                run_string += '--dist-flag {} '.format(config['gw-prior']['dist-flag'])

        run_string += '--eps-min {} '.format(config['kn-prior']['eps-min'])
        run_string += '--eps-max {} '.format(config['kn-prior']['eps-max'])

        if 'log-epsilon' in list_keys_in_prior:
            if (int (config['kn-prior']['log-epsilon'])) :
                run_string += '--log-eps0 '

        # check extra heating coeffs
        if 'extra-heat' in list_keys_in_prior:
            if (int (config['kn-prior']['extra-heat'])) :
                run_string += '--sample-heating '

        if 'heat-alpha' in list_keys_in_prior:
            run_string += '--heat-alpha {} '.format(config['kn-prior']['heat-alpha'])
        if 'heat-time' in list_keys_in_prior:
            run_string += '--heat-time {} '.format(config['kn-prior']['heat-time'])
        if 'heat-sigma' in list_keys_in_prior:
            run_string += '--heat-sigma {} '.format(config['kn-prior']['heat-sigma'])

        if 'tshift-max' in list_keys_in_prior:
            run_string += '--tshift-max {} '.format(config['kn-prior']['tshift-max'])
        else:
            logger.warning("Using default upper bound for time shift (1day). If the signal is not centered on the given data segment, this may create problems.")
            run_string += '--tshift-max {} '.format(86400.)

        if 'tshift-min' in list_keys_in_prior:
            run_string += '--tshift-min {} '.format(config['kn-prior']['tshift-min'])

    return run_string

def write_postproc_string(config, tags, outdir):
    """
        Write command string to execute bajes_postproc.py
        given a config file
    """

    pp_string = 'bajes_postproc.py  --outdir {} '.format(outdir)

    if 'gw' in tags:
        pp_string += '--spin-flag {} '.format(config['gw-prior']['spin-flag'])
        pp_string += '--tidal-flag {} '.format(config['gw-prior']['tidal-flag'])

        list_keys_in_pipe = np.transpose(list(config.items('pipe')))[0]
        if 'mpi' not in list_keys_in_pipe: config['pipe']['mpi'] = 0
        if config['pipe']['mpi']:
            pp_string += ' -n {} '.format(int(config['pipe']['nprocs'])/int(config['pipe']['nnodes']))
        else:
            pp_string += ' -n {} '.format(config['pipe']['nprocs'])

    return pp_string

if __name__ == "__main__":

    global logger

    confing_path = os.path.abspath(sys.argv[1])

    config = configparser.ConfigParser()
    config.optionxform = str
    config.sections()
    config.read(confing_path)

    # set output directory
    try:
        outdir = os.path.abspath(config['pipe']['outdir'])
        ensure_dir(outdir)

    except KeyError:
        raise BajesPipeError("Invalid or missing outdir in config file. Please specify the output directory in [pipe] section.")

    # set logger
    logger = set_logger(outdir=config['pipe']['outdir'], label='pipe')
    logger.info("Running bajes pipeline:")
    logger.info("... setting output directory ...")
    logger.info("  - {}".format(outdir))

    # check messengers
    try:
        tags = config['pipe']['messenger'].split(',')
        logger.info("... setting {} as messenger(s) ...".format(config['pipe']['messenger']))
    except KeyError:
        logger.error("Invalid or missing messenger in config file. Please specify the messengers (with comma-separated acronymes) in [pipe] section.")
        raise BajesPipeError("Invalid or missing messenger in config file. Please specify the messengers (with comma-separated acronymes) in [pipe] section.")

    # read tags
    for ti in tags:
        from bajes.obs import __knwon_messengers__
        if ti not in __knwon_messengers__:
            logger.error("Unknown messenger {}. Please use only knwon messengers: {}".format(ti,__knwon_messengers__))
            raise BajesPipeError("Unknown messenger {}. Please use only knwon messengers: {}".format(ti,__knwon_messengers__))

    # copy config
    with open(outdir+'/config.ini', 'w') as configfile:
        config.write(configfile)

    ini_string = ''
    if 'gw' in tags:

        from bajes.obs.gw import __known_events__, __known_events_metadata__

        try:
            ifos = config['gw-data']['ifos'].split(',')
            logger.info("... using {} IFOs ({}) ...".format(len(ifos), config['gw-data']['ifos']))
        except KeyError:
            logger.error("Invalid or missing ifos in config file. Please specify the IFOs\n(with comma-separated two-digit acronymes) in [data] section.")
            raise BajesPipeError("Invalid or missing ifos in config file. Please specify the IFOs\n(with comma-separated two-digit acronymes) in [data] section.")

        # parser for injection analysis
        if  config['gw-data']['data-flag'] == 'inject':
            logger.info("... writing links to injection files ...")
            ini_string  = write_inject_string(config, ifos, outdir)

        # parser for GWOSC data
        elif  config['gw-data']['data-flag'] == 'gwosc':

            try:
                if config['gw-data']['event'] in __known_events__:
                    logger.info("... writing urls to GWOSC archive with {} event ...".format(config['gw-data']['event']))

                    if config['gw-data']['event'] not in __known_events__:
                        logger.error("Impossible to read given event ({}), since it is not in the list of known events:\n{}.\nPlease specify an event from this list or use t-gps option.".format(config['gw-data']['event'], __known_events__))
                        raise BajesPipeError("Impossible to read given event ({}), since it is not in the list of known events:\n{}.\nPlease specify an event from this list or use t-gps option.".format(config['gw-data']['event'], __known_events__))

                    this_tgps = __known_events_metadata__[config['gw-data']['event']]['t_gps']
                    this_ifos = __known_events_metadata__[config['gw-data']['event']]['ifos']

                    # overwrite input t_gps
                    config['gw-data']['t-gps'] = '{}'.format( int (np.round(this_tgps)) )

                    for ifo in ifos:
                        if ifo not in this_ifos:
                            logger.warning("Warning: Selected ifo {} is not in the list of available detectors for this event ({}), the requested data does not exist and the code will fail.".format(ifo,this_ifos))

                else:
                    logger.info("... writing urls to GWOSC archive with GPS time ({}) ...".format(config['gw-data']['t-gps']))

            except KeyError:
                logger.error("... writing urls to GWOSC archive with GPS time ({}) ...".format(config['gw-data']['t-gps']))

            ini_string    = write_gwosc_string(config, ifos, outdir)

        # pass if data-flag is local
        elif  config['gw-data']['data-flag'] == 'local':
            pass

        # error otherwise
        else:
            logger.error("Invalid or missing data-flag in config file. Please use 'inject', 'local' or 'gwosc'")
            raise BajesPipeError("Invalid or missing data-flag in config file. Please use 'inject', 'local' or 'gwosc'")

    # fill core and postproc command
    logger.info("... writing executable ...")
    run_string      = write_run_string(config, tags, outdir)
    pp_string       = write_postproc_string(config, tags, outdir)

    # write executable
    execname        = write_executable(outdir, config, ini_string, run_string, pp_string)

    logger.info("... pipeline written in {}.".format(execname))

#!/usr/bin/env python
from __future__ import division, unicode_literals
import os
import numpy as np
import optparse as op

from bajes.pipe import set_logger, ensure_dir
from bajes.obs.gw.utils import read_gwosc

if __name__ == "__main__":

    parser=op.OptionParser()
    parser.add_option('--ifo',      dest='ifos',            type='string',  action="append", help='IFO tag, i.e. H1, L1, V1, K1, G1')

    parser.add_option('--event',    dest='event',           default=None,   help='GW event label')
    parser.add_option('--version',  dest='version',         default=None,   help='version of data')
    parser.add_option('--t-gps',    dest='t_gps',           type='float',   help='GPS time of the time series')
    parser.add_option('--srate',    dest='srate',           type='float',   help='sampling rate [Hz]')
    parser.add_option('--seglen',   dest='seglen',          type='float',   help='length of the segment [sec]')

    parser.add_option('-o','--outdir',default=None,type='string',dest='outdir',help='directory for output')
    (opts,args) = parser.parse_args()

    datadir = os.path.abspath(opts.outdir+'/data')
    ensure_dir(datadir)

    logger = set_logger(outdir=datadir, label='bajes_gwosc')
    logger.info("Running bajes gwosc-reading:")

    if opts.event and opts.t_gps is None:
        logger.info("... reading GPS time from event ...")
        from bajes.obs.gw import __known_events_metadata__ as known_events_metadata
        if opts.event not in list(known_events_metadata.keys()):
            logger.warning("Unable to identify event {}, this is not a known trigger".format(opts.event))
        opts.t_gps = int(known_events_metadata[opts.event]['t_gps'])
    elif opts.event is None and opts.t_gps:
        logger.info("... setting GPS time from input ...")
    else:
        logger.warning("Unable to set GPS time, please provide this information in the command line")

    GPSstart    = opts.t_gps - opts.seglen/2.
    GPSend      = opts.t_gps + opts.seglen/2.

    for ifo in opts.ifos:

        logger.info("... fetching data for {} ...".format(ifo))
        t , s = read_gwosc(ifo, GPSstart, GPSend, srate=opts.srate, version=opts.version)

        output_name = '/{}_STRAIN_{}_{}_{}.txt'.format(ifo , int(opts.seglen) , int(opts.srate), int(opts.t_gps))

        logger.info("... saving {} strain segment in {} ...".format(ifo , datadir + output_name))
        output_file = open(datadir + output_name, 'w')
        output_file.write('# GPStime \t strain \n')

        for ti,si in zip(t , s):
            output_file.write('{} \t {} \n'.format(ti,si))
        output_file.close()

    logger.info("... GWOSC data fetched.")

#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

from .strain import Series
from .detector import Detector
from .noise import Noise
from .waveform import Waveform

__known_events__ = ['GW150914','GW151012','GW151226','GW170104',
                    'GW170608','GW170729','GW170809','GW170814',
                    'GW170817','GW170818','GW170823']

__known_events_metadata__ = {'GW150914':    {'t_gps': 1126259462.4, 'ifos': 'H1,L1'},
                             'GW151012':    {'t_gps': 1128678900.4, 'ifos': 'H1,L1'},
                             'GW151226':    {'t_gps': 1135136350.6, 'ifos': 'H1,L1'},
                             'GW170104':    {'t_gps': 1167559936.6, 'ifos': 'H1,L1'},
                             'GW170608':    {'t_gps': 1180922494.5, 'ifos': 'H1,L1'},
                             'GW170729':    {'t_gps': 1185389807.3, 'ifos': 'H1,L1,V1'},
                             'GW170809':    {'t_gps': 1186302519.8, 'ifos': 'H1,L1,V1'},
                             'GW170814':    {'t_gps': 1186741861.5, 'ifos': 'H1,L1,V1'},
                             'GW170817':    {'t_gps': 1187008882.4, 'ifos': 'H1,L1,V1'},
                             'GW170818':    {'t_gps': 1187058327.1, 'ifos': 'H1,L1,V1'},
                             'GW170823':    {'t_gps': 1187529256.5, 'ifos': 'H1,L1'}}

__known_approxs__          = ['TaylorF2_3.5PN','TaylorF2_5.5PN',
                              'TaylorF2_5.5PN_7.5PNTides', 'TaylorF2_5.5PN_7.5PNTides2020',
                              'TaylorF2_5.5PN_3.5PNQM_7.5PNTides',
                              'TEOBResumS','TEOBResumSPA','HypTEOBResumS',
                              'NRPM', 'NRPM_ext', 'NRPM_ext_recal', 'NRPMw', 'NRPMw_recal',
                              'TEOBResumS_NRPM', 'TEOBResumSPA_NRPMw', 'TEOBResumSPA_NRPMw_recal',
                              'NRSur7dq4','NRHybSur3dq8','NRHybSur3dq8Tidal',
                              'MLGW', 'MLTEOBNQC', 'MLSEOBv4', 'MLGW-BNS', 'LALSimFD', 'LALSimTD']

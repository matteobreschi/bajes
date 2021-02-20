#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals
__import__("pkg_resources").declare_namespace(__name__)

def _get_git_hast():

    import os
    import sys
    import pathlib
    from . import __path__
    git_hash = 'UNKNOWN'

    # search in __path__ and sys.path for git folder
    for pi in __path__ + [di for di in sys.path if 'bajes' in di]:

        dir_path = os.path.abspath(os.path.join(pi,'..'))
        _ld = os.listdir(dir_path)

        if '.git' in _ld:
            git_dir = pathlib.Path(dir_path) / '.git'
            with (git_dir / 'HEAD').open('r') as head:
                ref = head.readline().split(' ')[-1].strip()
            with (git_dir / ref).open('r') as git_hash:
                git_hash = git_hash.readline().strip()
            break

    return git_hash

__version__ = '0.2.2'
__doc__     = "bajes [baɪɛs], Bayesian Jenaer Software. Python package for Bayesian inference developed at Friedrich-Schiller-Universtät Jena and specialized in the analysis of gravitational-wave and multi-messenger transients. The software is designed to be state-of-art, simple-to-use and light-weighted with minimal dependencies on external libraries. The source code is available at https://github.com/matteobreschi/bajes"
__githash__ = _get_git_hast()


# defining useful constant (SI)
MSUN_SI         = 1.9885469549614615e+30    # mass of the sun [kg]
MRSUN_SI        = 1476.6250614046494        # mass of the sun [m]
MTSUN_SI        = 4.925491025543576e-06     # mass of the sun [s]
CLIGHT_SI       = 299792458.0               # speed of light [m/s]
PARSEC_SI       = 3.085677581491367e+16     # 1 parsec [m]
GNEWTON_SI      = 6.67384e-11               # Newton constant [m^3 kg^-1 s^-2]
HPLANK_SI       = 6.626069578069735e-34     # Plank constant [m^2 kg / s]
KBOLTZMANN_SI   = 1.3806488056141758e-23    # Boltzmann constant [(m/s)^2 kg/K ]

# time and lenght conversions
PC_2_CM         = 3.085678e+18              #[cm/pc]
SEC_2_DAY       = 1.157407407e-5            #[day/s]
DAY_2_SEC       = 86400.                    #[sec/day]
SEC_2_HOUR      = 2.777778e-4               #[hr/s]
DAY_2_HOUR      = 24.                       #[hr/day]

from . import inf, obs

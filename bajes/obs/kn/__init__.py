#!/usr/bin/env python
__import__("pkg_resources").declare_namespace(__name__)

from .filter import Filter
from .lightcurve import Lightcurve

__photometric_bands__       = { 'B'         : 445e-9,
                                'Y'         : 1005e-9,
                                'U'         : 365e-9,
                                'g'         : 475e-9,
                                'z'         : 850e-9,
                                'V'         : 551e-9,
                                'W'         : 235e-9,
                                'J'         : 1220e-9,
                                'J1'        : 1055e-9,
                                'K'         : 2190e-9,
                                'Ks'        : 2150e-9,
                                'H'         : 1630e-9,
                                'R'         : 658e-9,
                                'I'         : 775e-9,
                                'F110W'     : 1162.4e-9,
                                'F160W'     : 1539.2e-9,
                                'F336W'     : 335.9e-9,
                                'F475W'     : 479.2e-9,
                                'F606W'     : 592.5e-9,
                                'F625W'     : 625.8e-9,
                                'F775W'     : 766.0e-9,
                                'F814W'     : 805.8e-9,
                                'F850W'     : 908.4e-9 }

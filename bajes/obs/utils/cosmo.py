from __future__ import division, unicode_literals, absolute_import
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from scipy.misc import derivative

class Cosmology(object):

    def __init__(self, cosmo=None, zmin=1.e-8, zmax=1.e2, ztol=1.e-8, maxfun=500, **kwargs):

        # import useful objects
        from astropy import cosmology
        self._z_at_value    = cosmology.z_at_value

        from ...pipe.utils import _get_astropy_version
        _av = _get_astropy_version()

        # initialize cosmology metric
        # if cosmo == None:
        #     logger.error("Unable to initialize cosmology class. Both cosmology name and keyword arguments are None.")
        #     raise RuntimeError("Unable to initialize cosmology class. Both cosmology name and keyword arguments are None.")
        if cosmo is not None:
            if int(_av[0])>=5:
                if cosmo not in cosmology.available:
                    logger.error("Unavailable cosmology model {}. Please use one of the followings:".format(cosmo, cosmology.available))
                    raise RuntimeError("Unavailable cosmology model {}. Please use one of the followings:".format(cosmo, cosmology.available))
                self.cosmo = getattr(cosmology, cosmo)
            else:
                logger.warning("Detected astropy version=={}. Reading {} attribute directly from astropy.cosmology module (unsafely).".format('.'.join(_av), cosmo))
                self.cosmo = getattr(cosmology, cosmo)
        else:
            logger.warning("Unspecified cosmological metric. The cosmological parameters are going to extracted from keyword args or switched to default settings (Planck18).")
            H0          = kwargs.get('H0',      67.7)
            Om0         = kwargs.get('Om0',     0.31)
            Tcmb0       = kwargs.get('Tcmb0',   2.725)
            Neff        = kwargs.get('Neff',    3.05)
            m_nu        = kwargs.get('m_nu',    [0., 0., 0.06])
            Ob0         = kwargs.get('Ob0',     0.049)
            self.cosmo  = cosmology.FlatLambdaCDM(H0, Om0, Tcmb0, Neff, m_nu, Ob0)
        # else:
            # logger.warning("Both cosmology name and keyword arguments are passed to Cosmology class. Ignoring keyword arguments.")
            # H0          = kwargs.get('H0',      70.)
            # Om0         = kwargs.get('Om0',     0.3)
            # Tcmb0       = kwargs.get('Tcmb0',   2.725)
            # Neff        = kwargs.get('Neff',    3.04)
            # m_nu        = kwargs.get('m_nu',    [0., 0., 0.])
            # Ob0         = kwargs.get('Ob0',     None)
            # self.cosmo  = cosmology.FlatLambdaCDM(H0, Om0, Tcmb0, Neff, m_nu, Ob0)

        # initialize arguments for z_at_value method
        self._zmax   = zmax
        self._zmin   = zmin
        self._ztol   = ztol
        self._maxfun = maxfun

    def vc_to_z(self,covol):
        return self._z_at_value(self.cosmo.comoving_volume, u.Quantity(covol, unit=u.Mpc*u.Mpc*u.Mpc),
                                zmin=self._zmin, zmax=self._zmax, ztol=self._ztol, maxfun=self._maxfun)

    def dl_to_z(self,dlum):
        return self._z_at_value(self.cosmo.luminosity_distance, u.Quantity(dlum, unit=u.Mpc),
                                zmin=self._zmin, zmax=self._zmax, ztol=self._ztol, maxfun=self._maxfun)

    def dl_to_vc(self,dlum):
        z = self._z_at_value(self.cosmo.luminosity_distance, u.Quantity(dlum, unit=u.Mpc),
                             zmin=self._zmin, zmax=self._zmax, ztol=self._ztol, maxfun=self._maxfun)
        return self.z_to_vc(z)

    def dc_to_z(self,dcom):
        return self._z_at_valuez(self.cosmo.comoving_distance, u.Quantity(dcom, unit=u.Mpc),
                                 zmin=self._zmin, zmax=self._zmax, ztol=self._ztol, maxfun=self._maxfun)

    def z_to_dl(self,z):
        return self.cosmo.luminosity_distance(z).value

    def z_to_vc(self,z):
        return self.cosmo.comoving_volume(z).value

    def z_to_dc(self,z):
        return self.cosmo.comoving_distance(z).value

    def dvc_dz(self, z, order=3):
        return derivative(self.z_to_vc, z, dx=self._ztol, order=order)

    def dz_ddl(self, dl, order=3):
        return derivative(self.dl_to_z, dl, dx=self._ztol*1e8, order=order)

    def dvc_ddl(self, dl, order=3):
        return derivative(self.dl_to_vc, dl, dx=self._ztol*1e8, order=order)

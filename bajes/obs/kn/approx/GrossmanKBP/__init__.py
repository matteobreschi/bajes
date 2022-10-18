from __future__ import division, unicode_literals, absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import numpy as np

from .model import KorobkinBarnesGrossmanPeregoEtAl, Shell, Heating, initialize_angular_axis, initialize_flux_factors

# Perego-Grossman-Korobkin-et-al. model
# * Nuclear heating rates described in Korobkin et al. (2012), arXiv:1206.2379
# * Ejecta expansion model described in Grossman et al. (2014), arXiv:1307.2943
# * Thermal efficiency described in Barnes et al. (2016), arXiv:1605.07218
# * Multi-component anisotropic model described in Perego et al. (2017), arXiv:1711.03982

#
# one-component wrappers
#

class korobkin_barnes_grossman_perego_et_al_isotropic_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 1
        self.components     = [Shell(name='isotropic',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_equatorial_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 1
        self.components     = [Shell(name='equatorial',  geom='equatorial',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_polar_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 1
        self.components     = [Shell(name='polar',  geom='polar',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

#
# two-component wrappers
#

class korobkin_barnes_grossman_perego_et_al_two_isotropic_isotropic_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 2
        self.components     = [Shell(name='isotropic1',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='isotropic2',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_two_isotropic_equatorial_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 2
        self.components     = [Shell(name='isotropic',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='equatorial',  geom='equatorial',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_two_isotropic_polar_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 2
        self.components     = [Shell(name='isotropic',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='polar',  geom='polar',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_two_equatorial_polar_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 2
        self.components     = [Shell(name='equatorial',  geom='equatorial',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='polar',  geom='polar',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_two_nrfit_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 2
        self.components     = [Shell(name='dyn',    geom='equatorial',   time=times,
                                     angles=angles, omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='wind',   geom='isotropic',   time=times,
                                     angles=angles, omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

#
# three-component wrappers
#

class korobkin_barnes_grossman_perego_et_al_three_isotropic_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 3
        self.components     = [Shell(name='isotropic1',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='isotropic2',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='isotropic3',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

class korobkin_barnes_grossman_perego_et_al_three_anisotropic_wrapper(KorobkinBarnesGrossmanPeregoEtAl):

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 3
        self.components     = [Shell(name='isotropic',  geom='isotropic',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='equatorial',  geom='equatorial',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v),
                               Shell(name='polar',  geom='polar',   time=times,
                                     angles=angles,     omegas=omegas,      heat=heat,
                                     v_min=v_min,   n_v=n_v)]

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

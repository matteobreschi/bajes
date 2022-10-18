from __future__ import division, unicode_literals
import numpy as np

def dereddening_CCM(wave, EBV=0.105, R_V=3.1):
    '''
        MAGNITUDE CORRECTION FOR DEREDDENING
        Input wavelength in nanometers
        - Input:
            wave 1D array (Nanometers)
            EBV: E(B-V) (default 0.105)
            R_V: Reddening coefficient to use (default 3.1)
        - Output:
            A_lambda correction according to the CCM89 Law.
    '''
    x = 1000./ wave  #Convert to inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ## Infrared ##
    mask = (x > 0.3) & (x < 1.1)
    if np.any(mask):
        a[mask] =  0.574 * x[mask]**(1.61)
        b[mask] = -0.527 * x[mask]**(1.61)

    ## Optical/NIR ##
    mask = (x >= 1.1) & (x < 3.3)
    if np.any(mask):
        xxx = x[mask] - 1.82
        # coefficients from O'O'Donnell (1994)
        c1 = [ 1. , 0.104,   -0.609,    0.701,  1.137, -1.718,   -0.827,    1.647, -0.505 ]
        c2 = [ 0.,  1.952,    2.908,   -3.989, -7.985, 11.102,    5.491,  -10.805,  3.347 ]
        a[mask] = np.poly1d(c1[::-1])(xxx)
        b[mask] = np.poly1d(c2[::-1])(xxx)

    ## Mid-UV ##
    mask = (x >= 3.3) & (x < 8.0)
    if np.any(mask):
        F_a = np.zeros_like(x[mask])
        F_b = np.zeros_like(x[mask])
        mask1 = x[mask] > 5.9
        if np.any(mask1):
            xxx = x[mask][mask1] - 5.9
            F_a[mask1] = -0.04473 * xxx**2 - 0.009779 * xxx**3
        a[mask] = 1.752 - 0.316*x[mask] - (0.104 / ( (x[mask]-4.67)**2 + 0.341 )) + F_a
        b[mask] = -3.090 + 1.825*x[mask] + (1.206 / ( (x[mask]-4.62)**2 + 0.263 )) + F_b

    ## Far-UV ##
    mask = (x >= 8.0) & (x < 11.0)
    if np.any(mask):
        xxx = x[mask] - 8.0
        c1 = [ -1.073, -0.628,  0.137, -0.070 ]
        c2 = [ 13.670,  4.257, -0.420,  0.374 ]
        a[mask] = np.poly1d(c1[::-1])(xxx)
        b[mask] = np.poly1d(c2[::-1])(xxx)

    #Now compute extinction correction
    A_V = R_V * EBV
    A_lambda = A_V * (a + b/R_V)
    return A_lambda

class Filter(object):

    def __init__(self, folder, lambdas, dered=True):

        self.lambdas    = lambdas
        self.magnitudes = {}
        self.mag_stdev  = {}
        self.times      = {}

        all_times  = []

        for k in list(self.lambdas.keys()):

            # read data from given folder
            # obs. the data file names have to be identical to the lambdas keys
            # obs. the data files should contain three columns, time, magnitudes and standard deviations
            try:
                t,  m,  sm          = np.genfromtxt(folder + '/{}.txt'.format(k), usecols=[0,1,2], unpack=True)
            except Exception as exc:
                raise RuntimeError("Error occured while loading {}".format(folder + '/{}.txt.'.format(k)))

            try:
                assert len(m) == len(sm)
                assert len(m) == len(t)
            except Exception as exc:
                raise RuntimeError("Unconsistent data length detected in magnitude file {}".format(folder + '/{}.txt.'.format(k)))

            self.magnitudes[k]  = m
            self.mag_stdev[k]   = sm
            self.times[k]       = t

            # apply dereddening, if requested
            if dered:
                self.magnitudes[k] -= dereddening_CCM(np.asarray([self.lambdas[k]]))

            all_times      = np.concatenate([all_times, t])

        self.all_times = np.sort(list(set(all_times)))

    @property
    def bands(self):
        return list(self.lambdas.keys())

    @property
    def wavelengths(self):
        return [self.lambdas[bi] for bi in self.bands]

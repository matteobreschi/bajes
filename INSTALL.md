# Installation

## Requirements

*bajes* is compatible with Python v3.7 or higher.
Before the installation, please check that the required Python modules are provided,

    astropy, numpy, scipy

If you are interested in using the provided pipeline for gravitational-wave and multi-messenger
transients, the software requires additional modules,

    h5py, mpi4py, corner, matplotlib, gwpy

and the samplers,

    cpnest, dynesty, emcee, ultranest

All these packages can be easily installed via `pip`.

## Installing this package

The *bajes* package is available on [`PyPI`](https://pypi.org/project/bajes/) and the installation can be performed using `pip` as

    pip install bajes

Alternatively, the source code can be found on [`GitHub`](https://github.com/matteobreschi/bajes)
and the package can be installed with `setuptools` routines,

    python setup.py install

 * *Note*, it is possible that your system will require the `sudo` specification
    in order to build the package. If this is the case, this option will prevent `setup.py` from estimating
    the path to the project repository. Then, the installation can be performed using the flag `-E`,

        sudo -E python setup.py install

Once *bajes* is installed, it is possible to perform Bayesian inference on arbitrary models,
see [`inf_tutorial`](https://github.com/matteobreschi/bajes/tree/release/v0.3.0/docs/inf_tutorial.ipynb).

The execution of the provided pipeline can be handled with a configuration file,
see [`conifg_example`](https://github.com/matteobreschi/bajes/tree/release/v0.3.0/docs/conifg_example.ini).
In order to execute the gravitational-wave pipeline routines,
the user should install additional packages depending on the requested template:
* [`TEOBResumS`](https://bitbucket.org/eob_ihes/teobresums)
* [`GWSurrogate`](https://pypi.org/project/gwsurrogate/)
* [`MLGW`](https://pypi.org/project/mlgw/)
* [`LALSuite`](https://lscsoft.docs.ligo.org/lalsuite/)
* [`JenpyROQ`](https://github.com/gcarullo/JenpyROQ)

# Installation

*bajes* is compatible with Python v3.7 or higher.
Before the installation, please check that the required Python modules are provided,

    astropy, numpy, scipy

If you are interested in using the provided pipeline for gravitational-wave and multi-messenger
transients, the software requires additional modules,

    h5py, mpi4py, corner, matplotlib, gwpy

and the samplers,

    cpnest, dynesty, emcee, ultranest

All these packages can be easily installed via `pip`.

The *bajes* installation is performed with `setuptools` running the command,

    python setup.py install

 * *Note*, it is possible that your system will require the `sudo` specification
    in order to build the package. If this is the case, this option will prevent `setup.py` from estimating
    the path to the project repository. Then, the installation can be performed using the flag `-E`,

        sudo -E python setup.py install

* *Note*, *bajes* is not currently listed in the [`PyPI`](https://pypi.org/), then the command `pip install bajes` is not available.
    Alternatively, the package can be installed using `pip`, running the following command inside the *bajes* repository,

        python -m pip install -U -e .

    We recommand to use the option `-e` in order to inlcude the project repository to `sys.path` allowing the system to identify your current git commit.
    We suggest to include also the `-U` option in order to upgrade all specified packages to the newest available version.

Once *bajes* is installed, it is possible to perform Bayesian inference on arbitrary models (see [`inf_tutorial`](docs/inf_tutorial.ipynb))
and to execute the provided pipeline with a configuration file (see [`conifg_example`](docs/conifg_example.ini)).

In order to execute the gravitational-wave pipeline routines,
the user should install some further packages:
* [`TEOBResumS`](https://bitbucket.org/eob_ihes/teobresums)
* [`GWSurrogate`](https://pypi.org/project/gwsurrogate/)
* [`MLGW`](https://pypi.org/project/mlgw/)
* [`LALSuite`](https://lscsoft.docs.ligo.org/lalsuite/)
* [`PyROQ-refactored`](https://github.com/bernuzzi/PyROQ)

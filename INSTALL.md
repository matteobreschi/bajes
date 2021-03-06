# Installation 

*bajes* is compatible with Python v3.7 or higher.
Before the installation, please check that the required Python modules are provided,

    numpy, scipy, astropy, mpi4py
    
If you are interested in using the provided pipeline for gravitational-wave and multi-messenger
transients, the software requires additional modules,

    gwpy, matplotlib, corner
    
an the samplers,

    cpnest, dynesty, emcee

All these packages can be easily installed via `pip`.

The *bajes* installation is performed with `setuptools` running the command,

    python setup.py install
    
 * *Note*, it is possible that your system will require the `sudo` specification
    in order to build the package. If this is the case, this option will prevent `setup.py` from estimating 
    the path to the project repository. Then, the installation can be performed using the flag `-E`,
    
        sudo -E python setup.py install
    
During this execution, the routine estimates the path of the *bajes*
repository, then be sure you are running the install from the  same directory.
Once *bajes* is installed, it is possible to perform Bayesian inference on arbitrary models (see [`inf_tutorial`](docs/inf_tutorial.ipynb))
and to execute the provided pipeline with a configuration file (see [`conifg_example`](docs/conifg_example.ini)).

Depending on the required gravitational-wave approximants,
the user should install some further packages:
* [`TEOBResumS`](https://bitbucket.org/eob_ihes/teobresums)
* [`GWSurrogate`](https://pypi.org/project/gwsurrogate/)
* [`MLGW`](https://pypi.org/project/mlgw/)
* [`LALSuite`](https://lscsoft.docs.ligo.org/lalsuite/) 



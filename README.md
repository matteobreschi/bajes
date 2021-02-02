<img src="docs/bajes.png" height=140>

*bajes* [baɪɛs] is a Python software for Bayesian inference developed at Friedrich-Schiller-Universtät Jena 
and specialized in the analysis of gravitational-wave and multi-messenger transients.
The software is designed to be state-of-art, simple-to-use and light-weighted 
with minimal dependencies on external libraries. 

## Installation

*bajes* is compatible with Python v3.7 (or higher)
and it is built on modules that can be easily installed via `pip`.
The mandatory dependencies are `numpy`, `scipy` and `astropy`.
However, the user might need to download some further packages.
See [`INSTALL`](INSTALL.md) for more information.

## Modules

*bajes* provides an homonymous Python module that includes:
* `bajes.inf`: implementation of the statistical objects and Bayesian workflow,
* `bajes.obs`: tools and methods for data analysis of multi-messenger signals.
For more details, visit [`gw_tutorial`](docs/gw_tutorial.ipynb).

## Inference

The *bajes* package  provides a user-friendly interface capable to easily set up a 
Bayesian analysis for an arbitrary model. Providing a prior file and a likelihood function, the command

    python -m bajes -p prior.ini -l like.py -o /path/to/outdir/
    
will run a parameter estimation job, inferring the properties of the input model.
For more details, visit [`inf_tutorial`](docs/inf_tutorial.ipynb) or type `python -m bajes --help`.

## Pipeline

The *bajes*  infrastructure allows the user to set up a pipeline for parameters 
estimation of multi-messenger transients. 
This can be easily done writing a configuration file,
that contains the information to be passed to the executables.
Subsequently,  the following command,

    bajes_pipe.py config.ini
    
will generates the requested output directory, if it does not exists, and 
the pipeline will be written into a bash executable (`/path/to/outdir/jobname.sub`). 
For more details, visit [`conifg_example`](docs/config_example.ini).

## Credits

*bajes* is developed by Matteo Breschi at the Friedrich-Schiller-Universität Jena with 
the contribution of Rossella Gamba and Sebastiano Bernuzzi.

If you find *bajes* useful in your research, please include the following citation in your publication,

    @article{Bajes:2021,
             author         = "Breschi, Matteo and Gamba, Rossella and Bernuzzi, Sebastiano",
             title          = "${\tt bajes}$: Bayesian inference of multimessenger astrophysical data, 
                              methods and application to gravitational-waves",
             eprint         = "2102.00017",
             archivePrefix  = "arXiv",
             primaryClass   = "gr-qc",
             month          = "1",
             year           = "2021"}
    
See [`CREDITS`](CREDITS.md) for more information.

## Acknowledgement

*bajes* has benefited from open source libraries, including the samplers,
* [`cpnest`](https://johnveitch.github.io/cpnest/)
* [`dynesty`](https://dynesty.readthedocs.io/)
* [`emcee`](https://emcee.readthedocs.io/)

and the gravitational-wave analysis packages,
* [`bilby`](https://lscsoft.docs.ligo.org/bilby/)
* [`gwbinning`](https://bitbucket.org/dailiang8/gwbinning/)
* [`lalsuite`](https://lscsoft.docs.ligo.org/lalsuite/) 
* [`pycbc`](https://pycbc.org)

We also acknowledge the LIGO-Virgo-KAGRA Collaboration for maitaining the [GWOSC archive](https://www.gw-openscience.org).

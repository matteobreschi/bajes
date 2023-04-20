## Changelog

# [v1.0.1] February 2023
* Include conversions for `.gwf` format
* Include options for frequency-domain injections

# [v1.0.0] October 2022
* Unify pipeline core with `__main__` routine
* Improve `Waveform` object and introduce `approx_dict`
* Improve `Lightcurve` object and introduce `approx_dict`
* Switch to `argparse` for option parsing
* Fix compatibility with `dynesty==1.2.3`

# [v0.3.0] June 2022
* Include ROQ approximation `JenpyROQ` in pipeline

# [v0.2.2] January 2022
* Introduce additional checks in `bajes_pipe.py`
* Improve `bajes_postproc.py`
* Include downsampling

# [v0.2.1] November 2021
* Reorganize `pipe/__init__.py`
* Introduce `gw/network.py`

# [v0.2.1] October 2021
* Introduce `NRPMw` model
* Improve posterior extraction for `dynesty`

# [v0.2.0] March 2021
* Introduce unified `SamplerBody`
* Introduce `ultranest` sampler
* Introduce ROQ approximation for `pipe` (work in progress)

# [v0.1.0] January 2021
* Reorganize pipeline in `pipe` module

# [v0.0.9] December 2020
* Introduce `__main__.py` interface
* Merge `TaylorF2` features branch
* Introduce mandatory dependencies in `setup.py`

# [v0.0.7] November 2020
* Reorganize repository introducing `inf` and `obs` modules
* Introduce MPI interface for  `bajes_parallel_core.py`

# [v0.0.6] September 2020
* Introduce `knmodule` and KN pipeline
* Include parametrized EOS and TOV solver in `gwmodule`

# [v0.0.5] July 2020
* Add `CHANGELOG.md`
* Add `LICENSE`
* Include `lmax` option in configuration file
* Include `lal`-waveforms wrapper
* Include options for `eccentricity` and `energy`-`angmom`
* Move to  `gwpy` for GW data fetching

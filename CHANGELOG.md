## Changelog

# [v0.2.2] July 2022
* Upload package to `PyPI`

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

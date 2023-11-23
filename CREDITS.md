# Credits

*bajes* has been developed by Matteo Breschi with the contribution of
Gregorio Carullo, Rossella Gamba, Ssohrab Borhanian, Emil Donkersloot and Sebastiano Bernuzzi.

If you use this software, please include the following [citation](https://arxiv.org/abs/2102.00017):

    @article{Bajes:2021,
             author = "Breschi, Matteo and Gamba, Rossella and Bernuzzi, Sebastiano",
             title = "{Bayesian inference of multimessenger astrophysical data: Methods and applications to gravitational waves}",
             eprint = "2102.00017",
             archivePrefix = "arXiv",
             primaryClass = "gr-qc",
             doi = "10.1103/PhysRevD.104.042001",
             journal = "Phys. Rev. D",
             volume = "104",
             number = "4",
             pages = "042001",
             year = "2021"
            }

The posterior samples computed with the *bajes* pipeline of the gravitational-wave events presented in GWTC-1 are available on [`Zenodo`](https://zenodo.org/record/4476594#.YBQcl3dKhQJ). If you use these data, please include the following citation:

    @misc{bajes_eob_catalog,
          author    = "Breschi, Matteo and Gamba, Rossella and Bernuzzi, Sebastiano",
          title     = "${\tt bajes}$: Bayesian inference of multimessenger astrophysical data,
                      methods and application to gravitational-waves",
          month     = jan,
          year      = 2021,
          publisher = {Zenodo},
          doi       = {10.5281/zenodo.4476594},
          url       = {https://doi.org/10.5281/zenodo.4476594}}

## Samplers

In order to perform the sampling, *bajes* relies on external libraries,
* `emcee`, the mcmc hammer, [github](https://github.com/dfm/emcee), [documentation](https://emcee.readthedocs.io/), [arxiv](https://arxiv.org/abs/1202.3665)
* `ptmcmc` is based on `ptemcee`, [github](https://github.com/willvousden/ptemcee), [documentation](https://ptemcee.readthedocs.io/en/stable/), [arxiv](https://arxiv.org/abs/1501.05823)
* `cpnest`, parallel nested sapling, [github](https://github.com/johnveitch/cpnest), [documentation](https://johnveitch.github.io/cpnest/)
* `dynesty`, dynamic nested sampling, [github](https://github.com/joshspeagle/dynesty), [documentation](https://dynesty.readthedocs.io/), [arxiv](https://arxiv.org/abs/1904.02180)
* `ultranest`, advanced nested sampling, [github](https://github.com/JohannesBuchner/UltraNest), [documentation](https://johannesbuchner.github.io/UltraNest/), [arxiv](https://arxiv.org/abs/2101.09604)

## Gravitational-wave pipeline

If you use one of the following gravitational-wave approximants, please provide the related citation,
* `TEOBResumS`: A. Nagar et al., [arxiv](https://arxiv.org/abs/1806.01772)
    * Precessing contributions are discussed in S. Akcay et al., [arxiv](https://arxiv.org/abs/2005.05338)
    * Eccentric model is described in D. Chiaramello and A. Nagar, [arxiv](https://arxiv.org/abs/2001.11736)
    * Template for hyperbolic captures is introduced in A. Nagar et al., [arxiv](https://arxiv.org/abs/2009.12857)
* `TEOBResumSPA`: R. Gamba et al., [arxiv](https://arxiv.org/abs/2012.00027)
* `HypTEOBResumS`: A. Nagar et al., [arxiv](https://arxiv.org/abs/2009.12857)
* `NRPM`: M. Breschi et al., [arxiv](https://arxiv.org/abs/1908.11418)
* `NRPMw`: M. Breschi et al., [arxiv](https://arxiv.org/abs/2205.09112)
* `MLGW`: S. Schmidt et al., [arxiv](https://arxiv.org/abs/2011.01958)
* `MLGW_BNS`: J. Tissino et al., (in preparation)
* `NRSur7dq4`: V. Varma et al., [arxiv](https://arxiv.org/abs/1905.09300)
* `NRHybSur3dq8`: V. Varma et al., [arxiv](https://arxiv.org/abs/1812.07865)
* `NRHybSur3dq8Tidal`: K. Barkett et al., [arxiv](https://arxiv.org/abs/1911.10440)
* `TaylorF2_5.5PN`: F. Messina et al., [arxiv](https://arxiv.org/abs/1904.09558)
    * `7.5PNTides` are taken from T. Damour et al., [arxiv](https://arxiv.org/abs/1203.4352)
    * `7.5PNTides2020` corresponds to Q. Henry et al., [arxiv](https://arxiv.org/abs/2005.13367)
* For the documentation of `LALSimulation` waveforms, please visit this [link](https://lscsoft.docs.ligo.org/lalsuite/)
* `JenpyROQ`: see documentation at this [link](https://github.com/GCArullo/JenpyROQ)

The *bajes* repository contains the official ASDs and the calibration envelopes
released with GWTC-1 (LVC, [arxiv](https://arxiv.org/abs/1811.12907)) and the design
ASDs for current and future detectors,
* LIGO Design sensitivity, LIGO Scientific Collaboration, [arxiv](https://arxiv.org/abs/1411.4547)
* Virgo Design sensitivity, F. Acernese et al., [arxiv](https://arxiv.org/abs/1408.3978)
* KAGRA Design sensitivity, T. Akutsu et al., [arxiv](https://arxiv.org/abs/1811.08079)
* Einstein Telescope (configuration D) sensitivity, S. Hild et al., [arxiv](https://arxiv.org/abs/1012.0908)
* Cosmic Explorer sensitivity, D. Reitze et al., [arxiv](https://arxiv.org/abs/1907.04833)

## Kilonova pipeline

If you use one of the following kilonova approximants, please provide the related citation,
* `GrossmanKBP`: D. Grossman et al., [arxiv](https://arxiv.org/abs/1307.2943)
    * Nuclear heating rates extracted from O. Korobkin et al., [arxiv](https://arxiv.org/abs/1206.2379)
    * Thermal efficiency described in J. Barnes et al., [arxiv](https://arxiv.org/abs/1605.07218)
    * Multi-component anisotropic model presented in A. Perego et al., [arxiv](https://arxiv.org/abs/1711.03982)

The *bajes* repository contains the bolometric magnitudes of AT2017gfo and the corresponding standard deviations
extracted from V. A. Villar et al., [arxiv](https://arxiv.org/abs/1710.11576)

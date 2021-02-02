# Credits

*bajes* has been developed by Matteo Breschi 
at Friedrich-Schiller-Universität Jena with the contribution of 
Rossella Gamba and Sebastiano Bernuzzi. 
If you use this software, please include the following citation:

    @article{Bajes:2021,
             author         = "Breschi, Matteo and Gamba, Rossella and Bernuzzi, Sebastiano",
             title          = "${\tt bajes}$: Bayesian inference of multimessenger astrophysical data, 
                              methods and application to gravitational-waves",
             eprint         = "2102.00017",
             archivePrefix  = "arXiv",
             primaryClass   = "gr-qc",
             month          = "1",
             year           = "2021"}

The posterior samples computed with the *bajes* pipeline of the gravitational-wave events presented in GWTC-1 are available on [`Zenodo`](https://zenodo.org/record/4476594#.YBQcl3dKhQJ). If you use these data, please include the following citation:

    @misc{bajes_eob_catalog,
          author    = "Breschi, Matteo and Gamba, Rossella and Bernuzzi, Sebastiano",
          title     = "${\tt bajes}$: Bayesian inference of multimessenger astrophysical data, 
                      methods and application to gravitational-waves",
          month     = jan,
          year      = 2021,
          publisher = {Zenodo},
          doi       = {10.5281/zenodo.4476594},
          url       = {https://doi.org/10.5281/zenodo.4476594}s}

In order to perform the sampling, *bajes* relies on external libraries,
* `cpnest`, parallel nested sapling, https://github.com/johnveitch/cpnest
* `dynesty`, dynamic nested sampling, https://dynesty.readthedocs.io/en/
* `emcee`, the mcmc hammer, https://github.com/dfm/emcee

If you use one of the following gravitational-wave approximants, please provide the related citation,
* `TEOBResumS`: A. Nagar et al., https://arxiv.org/abs/1806.01772
    * Precessing contributions are discussed in S. Akcay et al., https://arxiv.org/abs/2005.05338
    * Eccentric model is described in D. Chiaramello and A. Nagar, https://arxiv.org/abs/2001.11736
* `TEOBResumSPA`: R. Gamba et al., https://arxiv.org/abs/2012.00027
* `HypTEOBResumS`: A. Nagar et al., https://arxiv.org/abs/2009.12857
* `NRPM`: M. Breschi et al., https://arxiv.org/abs/1908.11418
* `MLGW`: S. Schmidt and W. Del Pozzo, https://arxiv.org/abs/2011.01958
* `NRSur7dq4`: V. Varma et al., https://arxiv.org/abs/1905.09300
* `NRHybSur3dq8`: V. Varma et al., https://arxiv.org/abs/1812.07865
* `NRHybSur3dq8Tidal`: K. Barkett et al., https://arxiv.org/abs/1911.10440
* `TaylorF2_5.5PN`: F. Messina et al., https://arxiv.org/abs/1904.09558 
    * `7.5PNTides` are taken from T. Damour et al., https://arxiv.org/abs/1203.4352
    * `7.5PNTides2020` corresponds to Q. Henry et al., https://arxiv.org/abs/2005.13367 
* For the documentation of `LALSimulation` waveforms, please visit https://lscsoft.docs.ligo.org/lalsuite/

The *bajes* repository contains the official ASDs and the calibration envelopes 
released with GWTC-1 (LVC, https://arxiv.org/abs/1811.12907)  and the design
ASDs for current and future detectors,
* LIGO Design sensitivity, LIGO Scientific Collaboration, https://arxiv.org/abs/1411.4547
* Virgo Design sensitivity, F. Acernese et al., https://arxiv.org/abs/1408.3978
* KAGRA Design sensitivity, T. Akutsu et al., https://arxiv.org/abs/1811.08079
* Einstein Telescope (configuration D) sensitivity, S. Hild et al., https://arxiv.org/abs/1012.0908
* Cosmic Explorer sensitivity, D. Reitze et al., https://arxiv.org/abs/1907.04833.pdf


# TO-DO-LIST

* Use `pkgutil.extend_path` to extend `__path__` variable
* Introduce interactive shell with `cmd.Cmd` (?! Maybe overkills)
* Unify constants in unit class (`bajes/obs/utils/units` ?!)
* Introduce timer in `logger`
* Fix `checkpoint_and_exit` in `Sampler`
* Replace `kwargs['case']` with `kwargs.get('case', default)`
* `inf`
   * Introduce hierarchical methods and error propagations
   * Introduce Gelman–Rubin convergence diagnostic for MCMC
   * Check (old) JointPrior with new Parameter settings
   * Check `ultranest`
      * Plots work only with resuming on
      * Checkpoint with ncpu = 1(serial) does not work (?!)
* `pipe`
   * Write pipeline for Condor (?)
   * Check Binning
   * Check PSD weights
   * Test of GR, e.g. PN tests
   * Parametrized EOS
   * ROQ : Emil / Greg
   * Include optional band-passing and padding
   * Sampler is not compatible with kn module ?!
   * Write pipeline for Condor (?)

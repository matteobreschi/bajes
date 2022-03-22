# Plan on how to restructure `bajes`

```
bajes/
    __init__.py                     mods: minors
    __main__.py                     mods: unify __main__ and core/parallel_core in single routine
    obs/                            mods: new module obs/gw/network.py
    inf/                            mods: minors
    pipe/                                               
        __init__.py                 mods: minors?
        gw_init.py                  mods: work with bajes_setup
        kn_init.py                  mods: work with bajes_setup
        log_likes.py                new: collection of gw, kn, and other log_like to be passed to the __main__
        bin/                        new: named scripts before
            bajes_inject            mods: mostly from script/bajes_inject.py
            bajes_gwosc             mods: mostly from script/bajes_read_gwosc.py
            bajes_pipe              new: prepare submission scripts (bash, slurm, condor)
            bajes_setup             new: prepare requested Likelihood and Prior objects
            bajes_postproc          mods: mostly from script/bajes_postproc.py
        run/                        new
            run.py                  mods: generalization from pipe/scripts/bajes_core.py used through __main__
            run_mpi.py              mods: generalization from pipe/scripts/bajes_parallel_core.py used through __main__
        utils/                      mods: mv utils/model.py to pipe/log_likes.py
```

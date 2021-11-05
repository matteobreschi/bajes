# plan on how to restructure `bajes`

```
bajes/
    __init__.py                     mods: ?
    __main__.py                     mods: ?  
    obs/                            mods: NEW obs/gw/network.py
    inf/                            mods: ?
    pipe/                                               
        __init__.py                 mods: ?
        gw_init.py                  mods: ?
        kn_init.py                  mods: ?
        log_likes.py                new: collection of gw, kn, and other log_like to be passed to the __main__
        bin/                        new: named scripts before
            bajes_inject            mods: mostly from script/bajes_inject.py
            bajes_gwosc             mods: mostly from script/bajes_read_gwosc.py
            bajes_make              new: prepare submission scripts (bash, slurm, condor) 
            bajes_setup             new: prepare correct log_like.py file for analysis, e.g. correct inputs and network for gw analysis
            bajes_postproc          mods: mostly from script/bajes_postproc.py
        run/                        new
            run_core.py             mods: generalization from pipe/scripts/bajes_core.py
            run_mpi.py              mods: generalization from pipe/scripts/bajes_parallel_core.py
        utils/                      mods: mv utils/model.py to pipe/log_likes.py
```

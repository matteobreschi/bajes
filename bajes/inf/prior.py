from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from .utils import dict_2_list

from collections import namedtuple
Variable    = namedtuple("Variable",    ("name","func","kwarg"),    defaults=(None,None,{}) )
Constant    = namedtuple("Constant",    ("name","value"),           defaults=(None,None) )

def draw_uniform_list(pr, N):
    Nparam = len(pr.names)
    init_samples = np.array([[np.random.uniform(pr.bounds[i][0],pr.bounds[i][1]) for i in range(Nparam)]  for _ in range(N)])
    return init_samples

def draw_uniform_samples(pr, Nmax):

    from .utils import list_2_dict

    Nparam       = len(pr.names)
    samples      = np.transpose([np.random.uniform(pr.bounds[i][0],pr.bounds[i][1], size=Nmax) for i in range(Nparam)])
    init_samples = [list_2_dict(samples[i], pr.names) for i in range(Nmax)]
    log_prob     = [pr.log_prior(init_samples[i]) for i in range(Nmax)]

    return init_samples, log_prob

def rejection_sampling(pr, Nmax, maxlogp=None):

    # draw initial points and compute (prior) probabilities
    init_samples, log_prob = draw_uniform_samples(pr, Nmax)

    if maxlogp == None:
        log_prob_max = np.max(log_prob)
    else:
        log_prob_max = np.max([maxlogp,np.max(log_prob)])

    acc         = ((log_prob - log_prob_max ) > np.log(np.random.uniform(0.,1.,size=len(log_prob))))
    Nout        = (acc).sum()
    samples     = np.array(init_samples)[acc]
    log_prob    = np.array(log_prob)[acc]

    return samples , log_prob, Nout

class Parameter(object):
    """
        Parameter object
    """

    def __init__(self,
                 name=None,
                 min=None, max=None,
                 prior='uniform',
                 periodic=0,
                 func=None, func_kwarg={},
                 interp_kwarg={'ngrid':2000, 'kind':'linear'},
                 **kwarg):
        """
            Initialize parameter

            Arguments:
            - name          : str, name of parameter
            - min           : float, lower bound
            - max           : float, upper bound
            - periodic      : bool, periodic (True) or reflective (False)
            - prior         : str, specify implemented prior distribution to be used:
                              uniform, log-uniform, linear, quadratic, power-law,
                              triangular, cosinusoidal, sinusoidal, exponential, exp-log,
                              normal, log-normal (default uniform)
            - func          : method, if not None this method will define the prior
                              distribution and the prior string will be ignored
            - func_kwarg    : dict, optional keyword arguments for input function
            - interp_kwarg  : dict, optional keyword arguments for interpolator,
                              the dictionary should contain two arguments:
                              - ngrid : int, number of point for the grid
                              - kind  : str, kind of interpolation for scipy.interpolate.interp1d
            - kwargs        : it is possible to pass further arguments depending on the
                              employed prior:
                              - deg : if power, deg is the power-law degree
                              - tau : if exponential, tau is the exponential decay factor
                              - mu, sigma : if normal, specify mean and stdev

        """

        # check name
        if name == None:
            raise AttributeError("Unable to initialize parameter, name is missing.")
        else:
            name = str(name)

        # check bounds
        if min == None or max == None:
            raise ValueError("Unable to initialize parameter {}, please define upper and lower bounds.".format(name))

        if min >= max :
            raise ValueError("Unable to initialize parameter {}, lower bound is greater or equal to the upper bound".format(name))

        bound = [float(min),float(max)]

        # check periodic
        periodic = int(periodic)

        # check/get distribution
        if func == None:

            from . import __known_probs__

            if not isinstance(prior, str):
                _all = ', '.join(__known_probs__)
                raise AttributeError("Unable to initialize parameter {}, prior argument is not a string. Please use one of the followings: {} or a customized function using the func argument".format(name,_all))

            if prior not in __known_probs__:
                _all = ', '.join(__known_probs__)
                raise AttributeError("Unable to initialize parameter {}, unknown prior argument. Please use one of the followings: {} or a customized function using the func argument".format(name,_all))

            # if func == None, read prior from prior string
            from .utils import get_parameter_distribution_from_string
            prob =  get_parameter_distribution_from_string(name, prior, min, max, kwarg)

        else:

            prior = 'custom'

            if callable(func):
                from .utils import initialize_param_from_func
                prob = initialize_param_from_func(name, min, max, func, kwarg=func_kwarg, **interp_kwarg)

            else:
                raise AttributeError("Unable to initialize parameter {}, requested probability function is not callable.".format(name))

        # set properties
        self._name      = name
        self._bound     = bound
        self._periodic  = periodic

        self._kind  = prior
        self._prob  = prob

    def __eq__(self, other):
        # check prior arguments
        args    = ['_name', '_bound', '_periodic', '_kind']
        bools   = [self.__dict__[ai] == other.__dict__[ai] for ai in args]
        # if customized prior, check agreement between interpolators
        if self._kind == 'custom' and bools[-1]:
            bools.append(all(np.abs(self._prob.log_density.x - other._prob.log_density.x) < 1e-30))
            bools.append(all(np.abs(self._prob.log_density.y - other._prob.log_density.y) < 1e-30))
        return all(bools)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        return self._name

    @property
    def bound(self):
        return self._bound

    @property
    def lower_bound(self):
        return self._bound[0]

    @property
    def upper_bound(self):
        return self._bound[1]

    @property
    def periodic(self):
        return self._periodic

    def log_density(self, x):
        return self._prob.log_density(x)

    def cumulative(self, x):
        return self._prob.cumulative(x)

    def quantile(self, x):
        return self._prob.quantile(x)

class Prior(object):
    """
        Prior object
    """
    def __init__(self, parameters, variables=[], constants=[]):
        """
            Initialize Prior object

            Arguments:
            - parameters  : list of bajes.inf.Parameter objects
            - variables   : list of bajes.inf.Variable objects, default empty
            - constants   : list of bajes.inf.Constant objects, default empty
        """

        self.ndim   = len(parameters)

        # reading constant properties
        self.const  = {ci.name : ci.value for ci in constants}

        # reading variables properties
        self.v_names    = []
        self.v_funcs    = []
        self.v_kwargs   = []

        for vi in variables:

            # check that every element in parameters is a Parameter object
            if not isinstance(vi, Variable):
                logger.error("The Prior received a variable that is not a Variable object.")
                raise ValueError("The Prior received a variable that is is not a Variable object.")

            # check that name is not in constants
            if vi.name in list(self.const.keys()):
                logger.error("Repeated name {} between variables and contants. Please use different names.".format(vi.name))
                raise ValueError("Repeated name {} in sampling variables and contants. Please use different names.".format(vi.name))

            # append information
            self.v_names.append(vi.name)
            self.v_funcs.append(vi.func)
            self.v_kwargs.append(vi.kwarg)

        # checking/reading parameters
        temp_names  = []

        for pi in parameters:

            # check that every element in parameters is a Parameter object
            if not isinstance(pi, Parameter):
                logger.error("The Prior received a parameter that is not a Parameter object.")
                raise ValueError("The Prior received a parameter that is is not a Parameter object.")

            # check bounds lengths
            if len(pi.bound) !=2:
                logger.error("Wrong prior bounds for {} parameter. Bounds array length is different from 2.".format(pi.name))
                raise ValueError("Wrong prior bounds for {} parameter. Bounds array length is different from 2.".format(pi.name))

            # check that name is not repeated
            if pi.name in temp_names:
                logger.error("Repeate name {} for different parameters. Please use different names.".format(pi.name))
                raise ValueError("Repeate name {} for different parameters. Please use different names.".format(pi.name))

            # check that name is not in constants
            if pi.name in list(self.const.keys()) or pi.name in self.v_names:
                logger.error("Repeated name {} between parameters/contants/variables. Please use different names.".format(pi.name))
                raise ValueError("Repeated name {} in sampling parameters/contants/variables. Please use different names.".format(pi.name))

        self.parameters = parameters

    @property
    def names(self):
        return [p.name for p in self.parameters]

    @property
    def bounds(self):
        return [p.bound for p in self.parameters]

    @property
    def periodics(self):
        return [p.periodic for p in self.parameters]

    def this_sample(self, p):
        """
            Fill parameters dictionary with constants and variables
        """
        # collect variables
        v = {n: f(**p,**k) for n,f,k in zip(self.v_names, self.v_funcs, self.v_kwargs)}
        # merge parameters, variables and constants
        return {**p, **v, **self.const}

    def log_prior(self, x):
        """
            Compute log-prior from parameter
        """
        if isinstance(x, (list,np.ndarray)):
            return sum([pi.log_density(xi) for pi, xi in zip(self.parameters, x)])
        else:
            # if x is not a list, it may be a dictionary. However some structures might not be pure dictionaries (i.e. cpnest.LivePoint),
            # then, the only requirement is that we can access to the values of the parameter as a dictionary
            return sum([pi.log_density(x[pi.name]) for pi in self.parameters])

    def prior_transform(self, u):
        """
            Transform uniform sample in prior sample
        """
        return np.array([pi.quantile(xi) for pi,xi in zip(self.parameters, u)])

    def cumulative(self, x, name=None):

        if isinstance(x, (float, int, np.float)):
            x = [x]

        if len(x) == self.ndim:
            return np.prod(list(map(lambda pi, xi: pi.cumulative(xi), self.parameters, x)))
        else:

            if name == None:
                raise AttributeError("Unable to estimate partial cumulative probability. Please include the names of the requested parameters.")

            indx = [i for i,pi in enumerate(self.paramters) if pi.name in name]
            return np.prod(list(map(lambda i, xi: self.parameters[i].cumulative(xi), indx, x)))

    @property
    def sample(self):
        u = np.random.uniform(0,1,size=self.ndim)
        return self.prior_transform(u)

    def get_prior_samples(self, n, **kwargs):
        return np.array([self.sample for _ in range(n)])

    def rejection_sampling(self, Nmax, maxlogpr=None):
        return rejection_sampling(self,Nmax,maxlogp=maxlogpr)

    def sample_uniform(self, N):
        return draw_uniform_list(self, N)

    def in_bounds(self, x):
        if isinstance(x, (list,np.ndarray)):
            return all([pi.bound[0]<=xi<=pi.bound[1] for xi,pi in zip(x,self.parameters)])
        else:
            # if x is not a list/array, it may be a dictionary.
            # However some structures might not be pure dictionaries (i.e. cpnest.LivePoint).
            # Then, the only requirement is that we can access to the values of the parameter as a dictionary
            return all([pi.bound[0]<=x[pi.name]<=pi.bound[1] for pi in self.parameters])

class JointPrior(Prior):

    def __init__(self, priors):

        pars        = []
        temp_names  = []

        v_names     = []
        v_funcs     = []
        v_kwargs    = []
        const       = {}

        for i,pr in enumerate(priors):

            # iterate on sampling parameters
            for pi in pr.parameters:

                # include name if not yet
                if pi.name not in temp_names:
                    temp_names.append(pi.name)
                    pars.append(pi)

                # else check agreement
                else:
                    iprev = temp_names.index(pi.name)
                    if pi != pars[iprev]:
                        logger.error("Unable to join prior distributions. Common parameter {} has been found, but their priors do not agree.".format(pi.name))
                        raise AttributeError("Unable to join prior distributions. Common parameter {} has been found, but their priors do not agree.".format(pi.name))

            # iterate on variables
            for ni,fi,ki in zip(pr.v_names, pr.v_funcs, pr.v_kwargs):

                if ni in temp_names:
                    logger.error("Unable to join prior distributions. Repeated name {} between parameters/contants/variables.".format(ni))
                    raise ValueError("Unable to join prior distributions. Repeated name {} between parameters/contants/variables.".format(ni))

                if ni in v_names:

                    iprev = v_names.index(ni)
                    if v_funcs[iprev] != fi:
                        logger.error("Unable to join prior distributions. Inconsistent methods for variable {}.".format(ni))
                        raise ValueError("Unable to join prior distributions. Inconsistent methods for variable {}.".format(ni))

                else:
                    v_names.append(ni)
                    v_funcs.append(fi)
                    v_kwargs.append(ki)

            # iterate on constant properties
            for k in list(pr.const.keys()):
                # if constant name not in const,
                # update dictionary
                if k not in list(const.keys()):
                    const[k] = pr.const[k]
                # else, check that the values are in agreement
                else:
                    if const[k] != pr.const[k]:
                        logger.error("Common constant property {} has different value between different priors.".format(k))
                        raise RuntimeError("Common constant property {} has different value between different priors.".format(k))

        # set parameters
        self.ndim       = len(temp_names)
        self.parameters = pars

        # set constants and variables
        self.const      = const
        self.v_names    = v_names
        self.v_funcs    = v_funcs
        self.v_kwargs   = v_kwargs

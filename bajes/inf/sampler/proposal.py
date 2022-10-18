from __future__ import division, unicode_literals, absolute_import
import numpy as np
from itertools import repeat

from scipy.stats import gaussian_kde

import logging
logger = logging.getLogger(__name__)

from ..utils import list_2_dict, move_in_bound_periodic, move_in_bound_reflective, reflect_skyloc_3dets, reflect_skyloc_2dets
from ...pipe import eval_func_tuple

from collections import namedtuple
ModelTuple = namedtuple("ModelTuple", ("map_fn", "compute_log_prob_fn", "random"))

def _init_proposal_methods(priors, props=None, **kwargs):

    # set default proposal weights
    if props == None:
        logger.debug("Using default proposal settings")
        props = {'eig': 25., 'dif': 25., 'str': 20.,
                 'wlk': 15., 'kde': 10., 'pri': 5.}

    # assign methods
    _prop_fn = {'eig': EigenProposal,   'dif':  DEProposal,
                'str': StretchProposal, 'kde':  KDEProposal,
                'wlk': WalkProposal,    'pri':  PriorProposal,
                'slc': SliceProposal,   'gwt':  GWTargetProposal}

    # fill kwargs with missing information
    kwargs['ndim']      = priors.ndim
    kwargs['priors']    = priors

    # check for gwt-targeted
    if 'gwt' in list(props.keys()):
        if 'dets' not in list(kwargs.keys()) or 'like' not in list(kwargs.keys()):
            del props['gwt']
            logger.warning("Requested GW-targeted proposal, but likelihood and detectors objects are not given. Option ignored.")

    # initialise proposals
    _proposals = []
    _weights   = []
    for ki in list(props.keys()):

        logger.debug("Setting {} proposal".format(ki))
        if ki == 'gwt':
            # check if gw-targeted proposal list is empty
            _temp_prop = _prop_fn[ki](**kwargs)
            if len(_temp_prop.proposal_list) > 0:
                _proposals.append(_temp_prop)
                _weights.append(props[ki])
        else:
            _proposals.append(_prop_fn[ki](**kwargs))
            _weights.append(props[ki])

    # rescale weights
    _weights = np.array(_weights)/sum(_weights)

    return _proposals, _weights

def random_walk(args):
    s, c, n = args
    Nc      = len(c)
    inds    = np.random.choice(Nc, n, replace=False)
    cov     = np.cov(np.transpose(c[inds]))
    q       = np.random.multivariate_normal(s, cov)
    return q

class WalkProposal(object):
    """
        A `Goodman & Weare (2010)
        <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "walk move" with
        parallelization as described in `Foreman-Mackey et al. (2013)
        <https://arxiv.org/abs/1202.3665>`_.
        :param s: (optional)
        The number of helper walkers to use. By default it will use all the
        walkers in the complement.
        """

    def __init__(self, subset = 25, **kwargs):
        self.subset = subset

    def get_proposal(self, s, c, p, model):
        try:
            c   = np.concatenate(c, axis=0)
            q   = list(model.map_fn(random_walk, zip(s,repeat(c), repeat(self.subset))))
            return np.array(q), np.zeros(len(q), dtype=np.float64)
        except Exception as e:
            logger.error("Cannot take a larger sample than population when 'replace=False'. Increase the number of samples.")
            raise ValueError("Cannot take a larger sample than population when 'replace=False'. Increase the number of samples.")

def stretching(args):
    s, c, a = args
    Ns = len(s)
    Nc = len(c)
    ri = np.random.randint(Nc)
    zz = ((a - 1.0) * np.random.rand() + 1) ** 2.0 / a
    return c[ri] - (c[ri] - s) * zz, (Ns - 1.0) * np.log(zz)

class StretchProposal(object):
    """
        A `Goodman & Weare (2010)
        <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
        parallelization as described in `Foreman-Mackey et al. (2013)
        <https://arxiv.org/abs/1202.3665>`_.
        :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)
        """

    def __init__(self, stretch=2.0, **kwargs):
        self.a = stretch

    def get_proposal(self, s, c, p, model):
        c       = np.concatenate(c, axis=0)
        q_f     = list(model.map_fn(stretching, zip(s,repeat(c),repeat(self.a))))
        q, fact = np.transpose(q_f)
        return q, fact


def differential_evolution(args):

    s,c,scale = args
    Nc      = list(map(len, c))
    inds    = np.random.randint(len(Nc), size=2)
    w       = np.array([c[j][np.random.randint(Nc[j])] for j in inds])

    np.random.shuffle(w)
    g       = np.concatenate(np.diff(w, axis=0) * scale)
    return np.array(s)+g

class DEProposal(object):
    """
        Move class for differential evolution proposal
        implemented following Nelson et al. (2013)
    """

    def __init__(self, ndim, gamma=None, **kwargs):

        self.g0 = gamma
        if self.g0 is None:
            self.g0 = 2.38 / np.sqrt(2*ndim)

    def get_scale(self):
        if np.random.randint(2)==0:
            return np.random.normal(0,self.g0)
        else:
            return 1.

    def get_proposal(self, s, c, p, model):
        q = list(model.map_fn(differential_evolution,  zip(s,repeat(c), repeat(self.get_scale()))))
        return np.array(q), np.zeros(len(q), dtype=np.float64)

def kde_sample(args):
    s, kde  = args
    q       = np.concatenate(kde.resample(1))
    factor  = kde.logpdf(s) - kde.logpdf(q)
    return q, factor[0]

class KDEProposal(object):
    """
       Move class from mixture of KDE estimated from past and current chain samples.
       This proposal requires a lot of walkers
    """

    def __init__(self, bw_method=None, **kwargs):
        self.bw_method  = bw_method

    def get_proposal(self, s, c, p, model):
        c   = np.concatenate(c, axis=0)
        kde = gaussian_kde(c.T, bw_method=self.bw_method)
        q_f = list(model.map_fn(kde_sample, zip(s,repeat(kde))))
        q, factor = np.transpose(q_f)
        return np.array(q), np.array(factor)

class GWTargetProposal(object):
    """
        Move class for GW parameters
    """

    def __init__(self, priors, dets, like, rnd=1e-5, **kwargs):

        self.names      = priors.names
        self.ndim       = len(self.names)

        self.factor     = rnd
        self.ifos       = list(dets.keys())

        self.inners     = None
        self.params     = None

        self.detsloc    = {}
        for k in dets.keys():
            self.detsloc[k] = dets[k].location

        if len(self.ifos) >= 3:
            diff1 = self.detsloc[self.ifos[1]] - self.detsloc[self.ifos[0]]
            diff2 = self.detsloc[self.ifos[2]] - self.detsloc[self.ifos[0]]
            self.refvec = np.cross(diff1, diff2)
            self.refvec /= np.sqrt(self.refvec[0]**2 + self.refvec[1]**2 + self.refvec[2]**2)
            self.reflect_skyloc_func    = reflect_skyloc_3dets

        elif len(self.ifos) == 2:
            self.refvec = self.detsloc[self.ifos[0]] - self.detsloc[self.ifos[1]]
            self.refvec /= np.sqrt(self.refvec[0]**2 + self.refvec[1]**2 + self.refvec[2]**2)
            self.reflect_skyloc_func    = reflect_skyloc_2dets

        self.proposal_list  = []
        self.proposal_prob  = []
        # list the parameters in names and get the proposals

        if 'ra' in self.names and 'dec' in self.names and len(self.ifos) > 1:
            self.proposal_list.append(self.move_skyloc)
            self.proposal_prob.append(4.)

        if 'distance' in self.names and 'cos_iota' in self.names:
            self.proposal_list.append(self.move_distiota)
            self.proposal_prob.append(2.)

#        if 'distance' in self.names :
#            self.proposal_list.append(self.gibbs_distance)
#            self.proposal_prob.append(2.)
#            self.inners = like.inner_prods
#            self.params = priors.this_sample

        if 'psi' in self.names and 'phi_ref' in self.names:
            self.proposal_list.append(self.move_psiphi)
            self.proposal_prob.append(1.)

        if 'psi' in self.names:
            self.proposal_list.append(self.move_psi)
            self.proposal_prob.append(1.)

        # rescale proposal_prob otherwise np.random.choice will complain
        self.proposal_prob = np.array(self.proposal_prob)/np.sum(self.proposal_prob)

        self.where = {}
        for n,i in zip(self.names,range(self.ndim)):
            self.where[n] = i

    def get_proposal(self, s, c, p, model):

        Ns = len(s)
        this_proposals  = model.random.choice(self.proposal_list, p=self.proposal_prob, size=Ns)
        q = list(model.map_fn(eval_func_tuple, zip(this_proposals, s, repeat(c))))
        q = np.array(q) + model.random.normal(0,self.factor, size=(Ns,self.ndim))

        return q, np.zeros(Ns, dtype=np.float64)

    def gibbs_distance(self, s, c):

        p           = self.params(list_2_dict(s,self.names))
        dh, hh, dd  = self.inners(p)

        dist_maxL   = hh * p['distance'] / dh
        u_maxL      = 1./dist_maxL
        u_std       = 1./ ( np.sqrt(hh) * p['distance'] )
        u_new       = np.random.normal( u_maxL, u_std )

        q   = np.array(s)
        q[self.where['distance']] = 1./u_new

        return q

    def move_psiphi(self, s, c):

        # initialize current sample
        q = np.array(s)

        alpha = np.random.uniform(0,3.*np.pi)

        if np.random.randint(2)==1:
            q[self.where['psi']] = alpha - q[self.where['phi_ref']]
        else:
            q[self.where['phi_ref']] = alpha - q[self.where['psi']]

        return q

    def move_psi(self, s, c):

        # initialize current sample
        q = np.array(s)

        q[self.where['psi']] = (q[self.where['psi']] + np.pi/2)%np.pi

        if 'phi_ref' in self.names:
            q[self.where['phi_ref']] = (q[self.where['phi_ref']] + np.pi)%(2*np.pi)

        return q

    def move_distiota(self, s, c):

        # move for distance - iota
        c_t         = np.transpose(np.concatenate(c))
        distiota_kde = gaussian_kde([ c_t[self.where['cos_iota']] , c_t[self.where['distance']] ])

        # initialize current sample
        q = np.array(s)
        vec = np.concatenate(distiota_kde.resample(1))

        q[self.where['cos_iota']]        = vec[0]
        q[self.where['distance']]    = vec[1]

        return q

    def move_skyloc(self, s, c):

        # initialize current sample
        q = np.array(s)

        # reflect sky location, according to
        # arXiv:0911.3820v2 [astro-ph.CO] (2010)
        ra_new , dec_new, tshift_new = self.reflect_skyloc_func(s[self.where['ra']],
                                                                s[self.where['dec']],
                                                                self.refvec,
                                                                self.detsloc[self.ifos[0]])

        q[self.where['ra']]             = move_in_bound_periodic(ra_new,0.,2*np.pi)
        q[self.where['dec']]            = move_in_bound_reflective(dec_new, -np.pi/2., np.pi/2)

        if 'time_shift' in self.names:
            q[self.where['time_shift']] += tshift_new

        return q

#    def move_3ifos(self, s, c):
#        """
#            Function for emcee RedBlueMove,
#            args:
#            s       = sample
#            c       = complement
#            random  = random function
#        """
#
#        # initialize current sample
#        q = np.array(s)
#
#        # reflect sky location, according to
#        # arXiv:0911.3820v2 [astro-ph.CO] (2010)
#        ra_new , dec_new, tshift_new = self.reflect_skyloc_func(s[self.where['ra']],
#                                                                s[self.where['dec']],
#                                                                self.refvec,
#                                                                self.detsloc[self.ifos[0]])
#
#        # move other extrinsic parameters
#        # according to arXiv:1402.0053v1 [gr-qc] (2014)
#        dist_new, iota_new, psi_new = project_all_extrinsic(self.dets,
#                                                            s[self.where['ra']],
#                                                            s[self.where['dec']],
#                                                            np.arccos(s[self.where['cos_iota']]),
#                                                            s[self.where['distance']],
#                                                            s[self.where['psi']],
#                                                            s[self.where['time_shift']],
#                                                            ra_new, dec_new,
#                                                            s[self.where['time_shift']]+tshift_new,
#                                                            self.priors.t_gps)
#
#        q[self.where['ra']]             = (ra_new)%(2*np.pi)
#        q[self.where['dec']]            = np.pi/2. - reflect_circular(np.pi/2 - dec_new,np.pi)%(np.pi)
#        q[self.where['time_shift']]     = s[self.where['time_shift']] + tshift_new
#
#        if np.isnan([dist_new, iota_new, psi_new]).any() or np.isinf([dist_new, iota_new, psi_new]).any():
#            pass
#        else:
#            q[self.where['distance']]    = dist_new
#            q[self.where['cos_iota']]        = np.cos((iota_new)%(np.pi))
#            q[self.where['psi']]         = (psi_new)%(np.pi)
#
#        return q

class SliceProposal(object):
    """
        Move class inpsired by Ensamble Slice Proposal,
        M. Karamanis & F. Beutler,
        arXiv:2002.06212v1 [stat.ML] (2020)
    """

    def __init__(self, ndim, mu=1, threshold=1500, **kwargs):

        self.ndim   = ndim

        # set properties for covariance method
        self.mu_cov         = mu
        self.mu_cov_list    = []

        self.Nc_cov         = 0
        self.Ne_cov         = 0

        self.iter_cov       = 0

        # set properties for differential method
        self.mu_dif         = mu
        self.mu_dif_list    = []

        self.Nc_dif         = 0
        self.Ne_dif         = 0

        self.iter_dif       = 0

        # set general properties
        self.threshold  = threshold

    # cycle over covariance and differential slicing methods
    def get_proposal(self, s, c, p, model):
        if model.random.randint(2)==0:
            return self.get_proposal_cov(s, c, p, model)
        else:
            return self.get_proposal_dif(s, c, p, model)

    # general slice proposal algorithm
    def slice_proposal(self , s, p, eta, model):

        Np      = len(p)
        logY    = p + np.log(model.random.uniform(0,1,size=Np))
        u       = model.random.uniform(0,1,size=Np)
        L       = -u
        R       = L+1.

        pl, bl  = model.compute_log_prob_fn(s-(L*eta.T).T)
        pr, br  = model.compute_log_prob_fn(s-(R*eta.T).T)

        q_Ne_Nc = list(model.map_fn(eval_func_tuple,
                                    zip(repeat(self.slicing),
                                        repeat(model.compute_log_prob_fn),
                                        s, eta, logY, L, R, pl, pr)))

        q,Ne,Nc = np.transpose(q_Ne_Nc)
        return np.array(q),Ne,Nc

    # propose one new sample
    def slicing(self, func, si, ei, logY, L, R, pl, pr):

        Ne = 0
        Nc = 0

        while logY < pl :
            L       -= 1
            pl, _  = func([si + L*ei])
            Ne      += 1

        while logY < pr :
            R       += 1
            pr, _   = func([si + R*ei])
            Ne      += 1

        while True:

            alpha   = np.random.uniform(L,R)
            qi      = si + alpha * ei
            pq, _   = func([qi])

            if logY < pq:
                break

            if alpha < 0 :
                L = alpha
                Nc +=1
            else :
                R = alpha
                Nc +=1

        return qi, Ne, Nc

    # methods for covariance slice proposal
    def tune_mu_cov(self):
        if self.Ne_cov == 0 or self.iter_cov > self.threshold:
            self.mu_cov = self.mu_cov
        else:
            self.mu_cov =  2*self.mu_cov*self.Ne_cov/(self.Ne_cov+self.Nc_cov)
        self.mu_cov_list.append(self.mu_cov)

    def update_mean_and_cov(self, c):
        c_t     = np.transpose(np.concatenate(c))
        mean    = np.array([np.mean(c_t[i]) for i in range(self.ndim)])
        cov     = np.cov(c_t)
        return mean, cov

    def direction_vector_cov(self, mean, cov):
        return self.mu_cov * np.random.multivariate_normal(mean, cov)

    def get_proposal_cov(self, s, c, p, model):

        Nc      = list(map(len, c))
        Ns = len(s)
        ndim = s.shape[1]
        q = np.empty((Ns, ndim), dtype=np.float64)

        mean, cov = self.update_mean_and_cov(c)

        if self.iter_cov > self.threshold:

            # if iteration > threshold
            # compute mu as the mean of the previous mus (last 10%) and propose new points
            self.mu_cov = np.median(self.mu_cov_list[len(self.mu_cov_list)*9//10:])
            eta         = np.array([self.direction_vector_cov(mean, cov) for _ in range(Ns)])
            L           = -model.random.uniform(0,1,size=Ns)
            R           = L+1.
            alpha       = model.random.uniform(L,R)
            q           = np.array(s)-(alpha*eta.T).T

        else:

            eta         = np.array([self.direction_vector_cov(mean, cov) for _ in range(Ns)])
            q, Ne, Nc   = self.slice_proposal(np.array(s), p, eta, model)
            self.Ne_cov += (Ne).sum()
            self.Nc_cov += (Nc).sum()

            self.tune_mu_cov()

        self.iter_cov +=1

        return q, np.zeros(Ns, dtype=np.float64)

    # methods for differential slice proposal
    def tune_mu_dif(self):
        if self.Ne_dif == 0 or self.iter_dif > self.threshold:
            self.mu_dif = self.mu_dif
        else:
            self.mu_dif =  2*self.mu_dif*self.Ne_dif/(self.Ne_dif+self.Nc_dif)
        self.mu_dif_list.append(self.mu_dif)

    def direction_vector_dif(self, c):
        c       = np.concatenate(c, axis=0)
        inds    = np.random.choice(np.arange(len(c)),  size=2, replace=False)
        return (np.array(c[inds[0]]) - np.array(c[inds[1]])) * self.mu_dif

    def get_proposal_dif(self, s, c, p, model):

        Nc      = list(map(len, c))
        Ns = len(s)
        ndim = s.shape[1]
        q = np.empty((Ns, ndim), dtype=np.float64)

        if self.iter_dif > self.threshold:

            # if iteration > threshold
            # compute mu as the mean of the previous mus (last 10%) and propose new points
            self.mu_dif = np.median(self.mu_dif_list[len(self.mu_dif_list)*9//10:])
            eta         = np.array([self.direction_vector_dif(c) for _ in range(Ns)])
            L           = -model.random.uniform(0,1,size=Ns)
            R           = L+1.
            alpha       = model.random.uniform(L,R)
            q           = np.array(s)-(alpha*eta.T).T

        else:

            eta         = np.array([self.direction_vector_dif(c) for _ in range(Ns)])
            q, Ne, Nc   = self.slice_proposal(np.array(s), p, eta, model)
            self.Ne_dif += (Ne).sum()
            self.Nc_dif += (Nc).sum()

            self.tune_mu_dif()

        self.iter_dif +=1

        return q, np.zeros(Ns, dtype=np.float64)

def eigen_sample(args):
    s, c, r, f = args
    cov = np.cov(np.transpose(c))
    eigw , eigvec = np.linalg.eig(cov)
    eigw = np.real(eigw)
    return np.array(s) + np.array(eigw[r] * eigvec[:,r] * f)

class EigenProposal(object):
    """
        Move class for covariance proposal
    """

    def __init__(self, ndim, cov=None, **kwargs):

        self.ndim = ndim
        self.cov = cov

        if self.cov is None:
            self.cov =  1e-3 * np.matrix([[np.random.randint(2) for i in range(self.ndim)] for j in range(self.ndim)])

        try:
            self.eigw , self.eigvec = np.linalg.eig(self.cov)
            self.eigw = np.real(self.eigw)

        except Exception:
            self.cov =  1e-3 * np.diag(np.ones(self.ndim))
            self.eigw , self.eigvec = np.linalg.eig(self.cov)
            self.eigw = np.real(self.eigw)

    def get_proposal(self, s, c, p, model):
        Ns      = len(s)
        ndim    = s.shape[1]
        r       = model.random.randint(0,ndim,size=Ns)
        f       = model.random.normal(0,1,size=Ns)
        q       = list(model.map_fn(eigen_sample, zip(s, repeat(np.concatenate(c)), r, f)))
        return np.array(q), np.zeros(Ns, dtype=np.float64)


class PriorProposal(object):
    """
        Move class for prior proposal
    """

    def __init__(self, priors, ngrid=100, kind='linear', **kwargs):

        from scipy.interpolate import interp1d

        self.ndim    = priors.ndim
        self.inv_cdf = []
        for pi in priors.parameters:
            ax  = np.linspace(0., 1., ngrid)
            qnt = np.array([pi.quantile(ai) for ai in ax])
            self.inv_cdf.append(interp1d(x=ax, y=qnt, bounds_error=True,  kind=kind))

    def prior_proposal(self, i):
        u = np.random.uniform(0.,1.,size=self.ndim)
        return np.array([fi(ui) for fi,ui in zip(self.inv_cdf, u)])

    def get_proposal(self, s, c, p, model):
        Ns  = len(s)
        q   = list(model.map_fn(eval_func_tuple, zip(repeat(self.prior_proposal), range(Ns))))
        return np.array(q), np.zeros(Ns, dtype=np.float64)

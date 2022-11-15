import logging, numpy as np, os

logger = logging.getLogger(__name__)

from scipy.special         import i0e
from scipy.interpolate     import interp1d
# from numba.core.decorators import njit
from itertools             import product, compress, repeat

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


# # This is an internal bajes implementation of an ROQ algorithm by Emil Donkersloot.
# class ROQ:
#     def __init__(self, wave, psd, params, accuracy, path):
#         """
#         - `wave`: bajes.obs.gw.Waveform object
#         - `psd`: power spectral density of a given detector (as np array)
#         - `params`: `ROQ_Params` object specifying the training set
#         - `accuracy`: desired accuracy of the ROQ
#         - `path`: path to directory where it will save the genreated values (in multiple files)
#         """
#         self.wave = wave
#         self.quad_points = self.wave.freqs # original quadrature points
#         self.df = 1. / self.wave.seglen
#         self.weights = 1. / psd
#
#         self.params, self.params_to_vary = self.read_ROQ_params(params.param)
#         self.param_distributions = {
#             "mchirp": lambda x, min, max: min*(max/min)**x
#         }
#
#         self.epsilon = accuracy # epsilon
#         self.path = path
#
#     def read_ROQ_params(self, params):
#         params_to_vary = []
#         for key, vals in params.items():
#             try:
#                 if len(vals) == 2:
#                     params_to_vary.append(key)
#                 else:
#                     print("Need exactly two values as a range.")
#             # len(float) raises TypeError
#             except TypeError:
#                 continue
#
#         intrinsic_params = [
#             "mchirp",
#             "q",
#             "s1_r",
#             "s1_theta",
#             "s1_phi",
#             "s2_r",
#             "s2_theta",
#             "s2_phi",
#             "lamdba1",
#             "lambda2"
#         ]
#
#         for p in params_to_vary:
#             if p not in intrinsic_params:
#                 print("WARNING: varying non-intrinsic parameter " + p + ". Is this intended?")
#
#         return params, params_to_vary
#
#     def change_distribution(self, param, dist):
#         """ change distribution from uniform to custom distribution
#
#         - `param`: parameter key of which to change distribution (e.g. 'mchirp')
#         - `dist`: new distribution of parameter. Has to be a function of the form `dist(x, min, max)`.
#             Instead of calling `np.random.rand()*(max-min) + min` to pick random value for `param` (corresponding to
#             a uniform distribution) `dist(np.random.rand(), min, max)` is called instead
#         """
#         # checking for errors
#         self.params[param]
#         dist(np.nan, np.nan, np.nan)
#
#         self.param_distributions[param] = dist
#
#     def check_if_path_is_valid(self):
#         from os.path import isdir
#
#         return isdir(self.path)
#
#     # static variables declaring filenames
#     FILE_ORIG_FREQS = "original_frequencies.csv"
#     FILE_LINEAR_BASIS = "basis_linear.csv"
#     FILE_LINEAR_NODES = "nodes_linear.csv"
#     FILE_LINEAR_NODES_INDS = "nodes_inds_linear.csv"
#     FILE_QUADRATIC_BASIS = "basis_quadratic.csv"
#     FILE_QUADRATIC_NODES = "nodes_quadratic.csv"
#     FILE_QUADRATIC_NODES_INDS = "nodes_inds_quadratic.csv"
#     def save_data(self, linear_basis, linear_vals, linear_inds, quadratic_basis, quadratic_vals, quadratic_inds):
#         """ saves all relevant ROQ data in files generated inside `self.path`
#         """
#         from os.path import join
#
#         np.savetxt(join(self.path, self.FILE_ORIG_FREQS), self.quad_points)
#         np.savetxt(join(self.path, self.FILE_LINEAR_BASIS), linear_basis)
#         np.savetxt(join(self.path, self.FILE_LINEAR_NODES), linear_vals)
#         np.savetxt(join(self.path, self.FILE_LINEAR_NODES_INDS), linear_inds, "%i")
#         np.savetxt(join(self.path, self.FILE_QUADRATIC_BASIS), quadratic_basis)
#         np.savetxt(join(self.path, self.FILE_QUADRATIC_NODES), quadratic_vals)
#         np.savetxt(join(self.path, self.FILE_QUADRATIC_NODES_INDS), quadratic_inds, "%i")
#
#     @staticmethod
#     def dump_to_json(d, file):
#         """ dump a dictionary into a json file
#         """
#         from json import dump
#         with open(file, "w") as f:
#             dump(d, f)
#
#     @staticmethod
#     def polar_to_cartesian(r, theta, phi):
#         """ convert polar coordinates (r, theta, phi) (can be `numpy.array`'s) to cartesian (x, y, z)
#         """
#         x = r * np.sin(theta) * np.cos(phi)
#         y = r * np.sin(theta) * np.sin(phi)
#         z = r * np.cos(theta)
#
#         return x, y, z
#
#     @staticmethod
#     def cartesian_to_polar(x, y, z):
#         """ (x, y, z) -> (r, theta, phi)
#         see https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
#         """
#         r = np.sqrt(x**2 + y**2 + z**2)
#
#         if r == 0.0:
#             return 0.0, 0.0, 0.0
#
#         theta = np.arccos(z / r)
#         phi = np.arctan2(y, x)
#
#         return r, theta, phi
#
#     def convert_spin_to_cartesian(self, param):
#         """ convert `param["s1/2_r/theta/phi"]` -> `param[s1/2x/y/z]`
#         """
#         s1x, s1y, s1z = ROQ.polar_to_cartesian(
#             param.pop("s1_r"),
#             param.pop("s1_theta"),
#             param.pop("s1_phi")
#         )
#         s2x, s2y, s2z = ROQ.polar_to_cartesian(
#             param.pop("s2_r"),
#             param.pop("s2_theta"),
#             param.pop("s2_phi")
#         )
#
#         param["s1x"] = s1x
#         param["s1y"] = s1y
#         param["s1z"] = s1z
#         param["s2x"] = s2x
#         param["s2y"] = s2y
#         param["s2z"] = s2z
#
#     def convert_spin_to_spherical(self, param):
#         """ convert `param[s1/2x/y/z]]` -> `param["s1/2_r/theta/phi"`
#         """
#         s1_r, s1_t, s1_p = ROQ.cartesian_to_polar(
#             param.pop("s1x"),
#             param.pop("s1y"),
#             param.pop("s1z")
#         )
#         s2_r, s2_t, s2_p = ROQ.cartesian_to_polar(
#             param.pop("s2x"),
#             param.pop("s2y"),
#             param.pop("s2z")
#         )
#
#         param["s1_r"] = s1_r
#         param["s1_theta"] = s1_t
#         param["s1_phi"] = s1_p
#         param["s2_r"] = s2_r
#         param["s2_theta"] = s2_t
#         param["s2_phi"] = s2_p
#
#     def produce_roq(self, linear_training_sizes=[10_000, 100_000, 1_000_000], dump_params_to_json=False):
#         """
#         call this function to produce the ROQ data and store the result in the folder
#         specified by `self.path`
#
#         - `linear_training_sizes` chosse the training space sizes the basis building algorithm should
#         traverse
#         - `dump_params_to_json` dump the chosen linear and quadratic parameters into a json file inside the
#         `self.path` folder (Debug option)
#         """
#         if not self.check_if_path_is_valid():
#             print("PATH IS NOT VALID!")
#             return
#
#         linear_basis, linear_EIM_vals, linear_EIM_inds, linear_params = self.create_linear_roq(linear_training_sizes)
#
#         # TODO: split into two functions linear basis and quadratic basis
#         # (makes it possible to recover from an error in quadratic basis)
#
#         if dump_params_to_json:
#             ROQ.dump_to_json(linear_params, self.path+"/linear_params.json")
#
#         linear_basis_for_quad, linear_params = self.build_training_space_from_params(linear_params, spin_coord_sys="cartesian")
#
#         # create quadratic basis
#         quadratic_basis, quadratic_EIM_vals, quadratic_EIM_inds, quadratic_params = self.create_quadratic_roq(linear_basis_for_quad, linear_params)
#
#         # two if statements, so that breaking off after the linear case still produces some output
#         if dump_params_to_json:
#             ROQ.dump_to_json(quadratic_params, self.path+"/quadratic_params.json")
#
#         self.save_data(linear_basis, linear_EIM_vals, linear_EIM_inds,
#                         quadratic_basis, quadratic_EIM_vals, quadratic_EIM_inds)
#
#     def __uniform(self, min, max):
#         return np.random.rand()*(max-min) + min
#
#     def pick_corners_from_ROQ_Param(self):
#         """ return list of dicts with the corners of the training space provided by a dict
#         built from the `ROQ_Param` object (i.e. self.params)
#         """
#         corners = list(product(*[self.params[p] for p in self.params_to_vary]))
#
#         corner_params = []
#         for corner in corners:
#             temp = self.params.copy()
#             for (i, p) in enumerate(self.params_to_vary):
#                 temp[p] = corner[i]
#             corner_params.append(temp)
#
#         return corner_params
#
#     def rand_param(self):
#         """ choose a random paramater in the training space
#         """
#         new_param = self.params.copy()
#
#         for key in self.params_to_vary:
#             try:
#                 new_param[key] = self.param_distributions[key](
#                     np.random.rand(),
#                     self.params[key][0],
#                     self.params[key][1]
#                 )
#             except KeyError:
#                 new_param[key] = self.__uniform(
#                     self.params[key][0],
#                     self.params[key][1]
#                 )
#
#         self.convert_spin_to_cartesian(new_param)
#
#         return new_param
#
#     def pick_and_check(self, basis):
#         new_param = self.rand_param()
#         # compute waveform
#         wave = self.normalize(self.compute_wave(new_param))
#
#         if self.approx_error(wave, basis) < self.epsilon:
#             return False
#         else:
#             return wave, new_param
#
#     def pick_and_build_training(self, basis, desired_size):
#         """
#         """
#         training_size = 0
#         failed_attempts = 0
#
#         basis = np.array(basis)
#
#         training_set = []
#         training_params = []
#
#         for i in range(desired_size):
#             attempt = self.pick_and_check(basis)
#             if attempt:
#                 training_size += 1
#                 training_set.append(attempt[0])
#                 training_params.append(attempt[1])
#             else:
#                 failed_attempts += 1
#                 if failed_attempts % 1000 == 0:
#                     print("Failed attempts up to: " + str(failed_attempts))
#
#         print("Building training set of size " + str(training_size) + " took " + str(failed_attempts) + " failed attempts")
#
#         if training_size == 0:
#             return False
#
#         return np.array(training_set), np.array(training_params)
#
#     def build_training_space_from_params(self, params, basis=None, spin_coord_sys="spherical"):
#         """ build the training space corresponding to `params`,
#         which is a list of dicts
#
#         - `basis`: if a basis is provided, the generated functions are projected onto that basis
#             and if it is approximated better than `self.epsilon` it is not included in the training space
#         - `spin_coord_sys`: depending on if the coordinates for spin are given in spherical or
#             cartesian, it will convert them to cartesian if necessary.
#             Spin 1 and Spin 2 need to be in the same coordinate system
#         """
#         training_space = []
#         training_params = []
#
#         for p in params:
#             if spin_coord_sys == "spherical":
#                 self.convert_spin_to_cartesian(p)
#
#             temp = self.normalize(self.compute_wave(p))
#
#             if basis is not None and self.approx_error(temp, basis) < self.epsilon:
#                 continue
#
#             training_space.append(temp)
#             training_params.append(p)
#
#         return np.array(training_space), training_params
#
#     def create_linear_roq(self, training_sizes):
#         """ creates the linear basis and DEIM points
#
#         returns:
#         - linear basis evaluated at `self.wave.freqs`
#         - linear EIM points (frequency points)
#         - linear EIM points (list of indices, s.t. `self.wave.freqs[linear_EIM_inds] == linear_EIM_vals)
#         - parameters of the basis function
#         """
#         # start with corners of parameter space as linear basis, as in PyROQ
#         corner_params = self.pick_corners_from_ROQ_Param()
#         # we have to update corner_params s.t. it used cartesian parameters
#         corner_funcs, corner_params = self.build_training_space_from_params(corner_params)
#         linear_basis, linear_params = self.rb_greedy(corner_funcs, corner_params)
#
#         # # possible pre selction loop comes here
#         # choose some list of params for pre-selection, call it `pre_params`
#         # pre_training_space = self.build_training_space_from_params(pre_params)
#         # linear_basis, pre_params_selected = self.rb_greedy(pre_training_space, pre_params)
#         # linear_params += pre_params_selected
#
#         for (cnt, k) in enumerate(training_sizes):
#             temp = self.pick_and_build_training(linear_basis, k)
#             if temp:
#                 training_space = temp[0]
#                 training_params = temp[1]
#             else:
#                 break
#
#             linear_basis, new_basis_params = self.rb_greedy(training_space, training_params, linear_basis)
#             linear_params += new_basis_params
#
#             print("Done with linear run " + str(cnt+1))
#             cnt += 1
#
#         linear_EIM_vals, linear_EIM_inds = self.DEIM(np.array(linear_basis).T)
#
#         return linear_basis, linear_EIM_vals, linear_EIM_inds, linear_params
#
#     def build_quad_training_space_from_params(self, index_pairs, linear_params, linear_basis, quad_basis=None):
#         """ for `index_pairs = [[i1, i1], [i1, i2], ...]` build training space
#         `linear_basis[i1]^* * linear_basis[i1], linear_basis[i1]^* * linear_basis[i2], ...`
#
#         if `quad_basis` is provided, only add those functions to the training_space for which
#         the approximation error is greater than `self.epsilon`
#         """
#         training_space = []
#         training_space_params = []
#         for ip in index_pairs:
#             temp = self.normalize(np.conj(linear_basis[ip[0]]) * linear_basis[ip[1]])
#
#             if quad_basis is not None and self.approx_error(temp, quad_basis) < self.epsilon:
#                 continue
#
#             training_space.append(temp)
#             training_space_params.append((linear_params[ip[0]], linear_params[ip[1]]))
#
#         return np.array(training_space), training_space_params
#
#     def pick_and_build_quad_training(self, quad_basis, desired_size, quad_training_whole, linear_basis, linear_params):
#         """
#         """
#         training_size = 0
#         failed_attempts = 0
#
#         training_set = []
#         training_params = []
#
#         quad_basis = np.array(quad_basis)
#
#         if desired_size > len(quad_training_whole):
#             desired_size = len(quad_training_whole)
#
#         rng = np.random.default_rng()
#         random_items = rng.choice(list(quad_training_whole.keys()), desired_size, replace=False)
#         for i in random_items:
#             # rng.choice creates a 2d matrix in this case, thus we need to turn it back into a tuple
#             del quad_training_whole[(i[0], i[1])]
#
#             attempt = self.normalize(np.conj(linear_basis[i[0]]) * linear_basis[i[1]])
#
#             if self.approx_error(attempt, quad_basis) < self.epsilon:
#                 failed_attempts += 1
#                 if failed_attempts % 1000 == 0:
#                     print("Failed attempts up to " + str(failed_attempts))
#                 continue
#
#             training_size += 1
#             training_set.append(attempt)
#             training_params.append((linear_params[i[0]], linear_params[i[1]]))
#
#         if training_size == 0:
#             return False
#
#         print("Built training set of size " + str(training_size) + " with " + str(failed_attempts) + " failed attempts")
#
#         return np.array(training_set), training_params
#
#     def create_quadratic_roq(self, linear_basis, linear_params):
#         """ creates the quadratic basis and DEIM points
#
#         returns:
#         - linear basis evaluated at `self.wave.freqs`
#         - linear EIM points (frequency points)
#         - linear EIM points (list of indices, s.t. `self.wave.freqs[linear_EIM_inds] == linear_EIM_vals)
#         - parameters of the basis functions
#         """
#         for p in linear_params:
#             self.convert_spin_to_spherical(p)
#
#         # the arrays `linear_basis` and `linear_params` are one-to-one, so knowing the indices is enough
#         assert len(linear_basis) == len(linear_params)
#         n = len(linear_params)
#         quad_training_whole = list(product(range(n), range(n))) # corresponds to F_n^2
#         # we mostly need this for lookup and removing stuff, a dict allows us to do this in O(1) cost
#         quad_training_whole = dict.fromkeys(quad_training_whole, 0)
#
#         # possible preselction loop. c.f. `ROQ.create_linear_roq`
#
#         # first element chosen randomly
#         quad_basis = [self.normalize(np.conj(linear_basis[0]) * linear_basis[0])]
#         quad_params = [(linear_params[0], linear_params[0])]
#         # self.pick_and_build_quad_training will compare remaining size to np.inf and will take care
#         # of the rest
#         training_size = [10000, 100000, np.inf]
#         for (cnt, k) in enumerate(training_size):
#             temp = self.pick_and_build_quad_training(quad_basis, k, quad_training_whole, linear_basis, linear_params)
#             if temp:
#                 training_space = temp[0]
#                 training_params = temp[1]
#             # we didn't find any function outside our span but we didn't look at all functions yet
#             elif len(quad_training_whole) > 0:
#                 continue
#             # we looked at every function
#             else:
#                 break
#
#             quad_basis, new_quad_params = self.rb_greedy(training_space, training_params, quad_basis)
#             quad_params += new_quad_params
#
#             print("Done with quadratic run " + str(cnt+1))
#
#         quad_EIM_vals, quad_EIM_inds = self.DEIM(np.array(quad_basis).T)
#
#         return quad_basis, quad_EIM_vals, quad_EIM_inds, quad_params
#
#     def compute_wave(self, param):
#         # .copy(), because "mtot" messes other stuff up
#         return self.wave.compute_hphc(param.copy()).plus
#
#     def norm(self, func):
#         """ norm of function, using standard weights
#         """
#         return np.sqrt(np.abs(inner_product(self.weights, func, func, self.df)))
#
#     def normalize(self, func):
#         """ return copy of func / norm(func), with standard weights for norm
#         """
#         return func / self.norm(func)
#
#     def approx_error(self, func, basis):
#         """ approximation error
#
#         sigma = |func - P(func)|
#
#         where P is projection onto basis
#         """
#         return self.norm(func - ROQ.project(func, basis, self.weights, self.df))
#
#     @staticmethod
#     # @njit
#     def project(func, basis, weights, df):
#         """ project `func` onto `basis`
#         """
#         projection = np.zeros_like(func, dtype=func.dtype)
#         for e in basis:
#             projection += inner_product(weights, e, func, df) * e
#
#         return projection
#
#     def init_rb_greedy(self, training_space, training_space_params, prebuilt_basis):
#         if len(prebuilt_basis) == 0:
#             n = 0
#             mu = [ training_space_params[0] ] # arbitrary choice, training_set does not have to be sorted
#             e = [ training_space[0] ]
#             e[0] = self.normalize(e[0]) # normalize basis elements
#             added_elements = [0]
#         else:
#             n = len(prebuilt_basis) - 1
#             mu = []
#             e = [] + prebuilt_basis
#             added_elements = []
#         projected_funcs = np.zeros((len(training_space), len(training_space[0])), dtype=np.complex_)
#
#         for element in e:
#             self.update_projection(projected_funcs, element, training_space)
#
#         return n, mu, e, added_elements, projected_funcs
#
#     def update_projection(self, projection, new_basis, func_space):
#         """ updates projection by adding the projection of `func_space` onto the `new_basis` to `projection`
#         """
#         for f in range(len(func_space)):
#             projection[f] += inner_product(self.weights, new_basis, func_space[f], self.df) * new_basis
#
#         # why is the second method slower, when I am able to eliminate a for loop??
#         # tested with around 8000 x points and 10 000 functions in func_space:
#         # for 10 steps, above: 8.5s, below: 12.2s
#         # in_prod = lambda f: self.inner_product(self.weights, new_basis, f, self.df)
#
#         # projection += np.outer(np.apply_along_axis(in_prod, 1, func_space), new_basis)
#
#     def rb_greedy(self, training_space, training_space_params, prebuilt_basis=[]):
#         """ Reduced basis greedy algorithm as outlined in Alg. 1 and 4 of arXiv:1210.0577v2
#         - `training_space`: matrix consisting of the training space we use to build the basis
#         - `training_space_params`: corresponding parameters
#         - `prebuilt_basis=[]`: prebuilt basis which should be expanded upon
#
#         returns:
#         - reduced basis (including the prebuilt one)
#         - parameters used to built (or expand) the reduced basis from the training space
#         """
#         # naming conventions follow Antil et al.
#
#         # NOTE: to make it more efficient the approximation error used here (projection error)
#         # is pretty much hardcoded into this function (see `projected_funcs` and `self.update_projection`
#         # and also how `sigmas` is computed).
#         # If another error function was to be used (like the DEIM error used in PyROQ), then this function
#         # needs to be updated a bit more thoroughly
#         # For all other pieces of code one just has to update the function `self.approx_error`
#
#         assert len(training_space) == len(training_space_params)
#
#         # if we start out with 10 000 functions, it sounds reasonable to realloc after 1000
#         # are already approximated well. If we have 1000 functions or less it's fast enough without
#         # Value can be tweaked
#         if len(training_space) > 1000:
#             realloc_if_keep_over = len(training_space) // 10
#         else:
#             realloc_if_keep_over = False
#
#         sigma = np.inf
#         # offload work to increase readability
#         n, mu, e, added_elements, projected_funcs = self.init_rb_greedy(
#             training_space,
#             training_space_params,
#             prebuilt_basis
#         )
#
#         while True:
#             n += 1
#             sigmas = np.array([self.norm(training_space[i] - projected_funcs[i]) for i in range(len(training_space))]) # largest error for current subspace e
#             index_max = np.argmax(sigmas)
#             sigma = sigmas[index_max]
#
#             print("Greedy Error (mu_n+1): " + str(sigma) + ", with basis length (n): " + str(len(e)))
#             if sigma <= self.epsilon:
#                 break
#
#             if index_max in added_elements:
#                 print("Accuracy couldn't be reached.")
#                 print("Breaking off at error: " + str(sigmas[added_elements[-1]]))
#                 break
#             else:
#                 added_elements.append(index_max)
#
#             mu.append(training_space_params[index_max])
#             e.append(training_space[index_max] - projected_funcs[index_max]) # project onto old basis
#             e[n] = self.normalize(e[n])
#             self.update_projection(projected_funcs, e[n], training_space)
#
#             # remove well approximated functions
#             if realloc_if_keep_over:
#                 keep = sigmas > self.epsilon
#                 if len(sigmas) - keep.sum() > realloc_if_keep_over:
#                     print("Starting to remove " + str(len(sigmas) - keep.sum()) + " elements")
#                     # training_space_params is a list
#                     training_space_params = list(compress(training_space_params, keep))
#                     # training_space and projected_fucns are numpy arrays
#                     training_space = training_space[keep]
#                     projected_funcs = projected_funcs[keep]
#
#                     # already added elements are sure to be removed, thus we can reset `added_elements`
#                     added_elements = []
#
#                     print("Done removing")
#
#         return e, mu
#
#     def DEIM(self, V):
#         """ Algorithm 5 from Antil, et al.
#
#         V: (Mxm)-matrix as defined in paper. Column vectors fo V must be linearly independent
#            i.e. (e_1(x_1) e_2(x_1) ... e_m(x_1)) \n
#                 (e_1(x_2) e_2(x_2) ... e_m(x_2)) \n
#                 ( ... )
#
#         return: - interpolation points (ndarray)
#                 - interpolation points index positions (ndarray dtype=int)
#         """
#         # already bring them into the correct final shape to avoid costly appending
#         j = np.argmax(np.abs(V[:, 0])) # j in [0, M)
#         U = np.zeros(np.shape(V), dtype=np.complex128) # Mxm matrix
#         P = np.zeros(np.shape(V)) # Mxm matrix
#         p_xval = np.zeros(np.shape(V)[1], dtype=self.quad_points.dtype)
#         p_ind = np.zeros(np.shape(V)[1], dtype=np.int_) # m vector
#
#         # init U, P, pm
#         U[:, 0] = V[:, 0] # e_1 as first column
#         P[j, 0] = 1 # unit column vector with single unit entry at index j
#         p_xval[0] = self.quad_points[j] # actual x values
#         p_ind[0] = j # index positions of the x values
#
#         for i in range(1, len(p_ind)): # from 1 to m-1
#             c = np.linalg.solve(P[:, :i].transpose() @ U[:, :i], P[:, :i].transpose() @ V[:, i]) # solve (P^T . U) c = (P^T) e_1 for c
#             r = V[:, i] - U[:, :i] @ c
#             j = np.argmax(np.abs(r))
#             U[:, i] = r
#             P[j, i] = 1 # we only want unit columns, the rest is already 0
#             p_ind[i] = j
#             p_xval[i] = self.quad_points[j]
#             print("Ready with i=" + str(i) + "/" + str(len(p_ind) - 1))
#
#         return p_xval, p_ind
#
# # because calling a static method inside another static method does not compile with Numba,
# # offload this function to outside the class
# # @njit
# def inner_product(weights, func_1, func_2, df):
#     """ discrete overlap integral
#         (a,b) = 4 * df * Re[sum_i w_i * a(f_i)^* * b(f_i)]
#     where:
#         - df = 1 / seglen
#         - {w_i} = weights (typically 1 / PSD)
#         - a = func_1
#         - b = func_2
#     """
#     return 4. * df * np.real((weights * np.conj(func_1) * func_2).sum())
#
# class ROQ_Params:
#     def __init__(self,
#         mchirp      = 1,
#         q           = 1,
#         s1_r        = 0,
#         s1_theta    = 0,
#         s1_phi      = 0,
#         s2_r        = 0,
#         s2_theta    = 0,
#         s2_phi      = 0,
#         lambda1     = 0,
#         lambda2     = 0,
#         distance    = 400.,
#         iota        = np.pi,
#         ra          = 1.7,
#         dec         = -0.75,
#         psi         = 0,
#         time_shift  = 0,
#         phi_ref     = 0,
#         f_min       = 20,
#         f_max       = 1024,
#         srate       = 2048,
#         seglen      = 16,
#         tukey       = 0.1,
#         t_gps       = 1,
#         lmax        = 0,
#         eccentricity = 0
#     ):
#         """ Parameter object for the `ROQ` class. Initialize standard values common to *all* functions
#         in the training space. Specify the training set by using `ROQ_Params.set_range(..)` on your
#         `ROQ_params` object.
#         """
#         self.param = {
#             'mchirp'     : mchirp,        # chirp mass [solar masses]
#             'q'          : q,             # mass ratio
#             's1_r'       : s1_r,
#             's1_theta'   : s1_theta,
#             's1_phi'     : s1_phi,
#             's2_r'       : s2_r,
#             's2_theta'   : s2_theta,
#             's2_phi'     : s2_phi,
#             'lambda1'    : lambda1,     # primary tidal parameter
#             'lambda2'    : lambda2,     # secondary tidal parameter
#             'distance'   : distance,    # distance [Mpc]
#             'iota'       : iota,        # inclination [rad]
#             'ra'         : ra,          # right ascension [rad]
#             'dec'        : dec,         # declination [rad]
#             'psi'        : psi,         # polarization angle [rad]
#             'time_shift' : time_shift,  # time shift from GPS time [s]
#             'phi_ref'    : phi_ref,     # phase shift [rad]
#             'f_min'      : f_min,       # minimum frequency [Hz]
#             'f_max'      : f_max,       # maximum frequency [Hz]
#             'srate'      : srate,       # sampling rate [Hz]
#             'seglen'     : seglen,      # segment duration [s]
#             'tukey'      : tukey,       # parameter for tukey window
#             't_gps'      : t_gps,       # GPS trigger time
#             'lmax'       : lmax,        # maximum l of higher-order modes (set to 0 for l=2,m=2 only)
#             'eccentricity' : eccentricity, # eccentricity (from 0 to 1)
#         }
#
#     def set_range(self, parameter, min, max):
#         """ set a range for a parameter to be used in the ROQ training
#         """
#         # dict[key] = ... does not raise a `KeyError` if key is not present
#         # dict[key] does
#         self.param[parameter]
#         self.param[parameter] = [min, max]
#
#
# # This class calls an internal bajes implementation of an ROQ algorithm.
# class Initialise_bajes_ROQ_for_inference:
#
#     def __init__(self, path):
#         """
#         `ROQ_handle` handles the output of the main ROQ module. These are stored in different files
#         in a folder specified in the `ROQ` module.
#
#         Parameters:
#         - `path` path to folder containing the ROQ output
#         """
#         self.__read_in_file(path)
#
#         # inverse of L tilde
#         self.linear_basis_at_linear_nodes_inv = np.linalg.inv(self.linear_basis[:, self.linear_nodes_ind])
#         # inverse of Q tilde
#         self.quad_basis_at_quad_nodes_inv = np.linalg.inv(self.quadratic_basis[:, self.quadratic_nodes_ind])
#
#     def __read_in_file(self, path):
#         """ reads in an ROQ file
#         returns:
#         - frequency nodes the ROQ was trained on
#         - linear roq nodes
#         - linear basis (evaluated at original frequencies)
#         - quadratic roq nodes
#         - quadratic basis (evaluated at original frequencies)
#         """
#         # we assume the bases are in the format e.g. (B_j(f_i))_ji, i.e. j is the row and i the column
#         from os.path import join
#
#         self.orig_freqs = np.loadtxt(join(path, ROQ.FILE_ORIG_FREQS))
#         self.linear_basis = np.loadtxt(join(path, ROQ.FILE_LINEAR_BASIS), dtype=np.complex_)
#         self.linear_nodes = np.loadtxt(join(path, ROQ.FILE_LINEAR_NODES))
#         self.linear_nodes_ind = np.loadtxt(join(path, ROQ.FILE_LINEAR_NODES_INDS), dtype=np.int_)
#         self.quadratic_basis = np.loadtxt(join(path, ROQ.FILE_QUADRATIC_BASIS), dtype=np.complex_)
#         self.quadratic_nodes = np.loadtxt(join(path, ROQ.FILE_QUADRATIC_NODES))
#         self.quadratic_nodes_ind = np.loadtxt(join(path, ROQ.FILE_QUADRATIC_NODES_INDS), dtype=np.int_)
#
#         # make sure everything is fine
#         assert len(self.orig_freqs) == len(self.linear_basis[0])
#         assert len(self.orig_freqs) == len(self.quadratic_basis[0])
#         assert len(self.linear_basis) == len(self.linear_nodes)
#         assert len(self.quadratic_basis) == len(self.quadratic_nodes)
#         assert len(self.linear_nodes) == len(self.linear_nodes_ind)
#         assert len(self.quadratic_nodes) == len(self.quadratic_nodes_ind)
#
#     # ? depending on how the inner product is defined in bajes, maybe pull the 4*df out
#     def compute_linear_weights_bajes(self, data, psd, time_coalescence):
#         """ compute the linear weights according to Eq. TODO of my own file
#         """
#         assert len(data) == len(self.orig_freqs)
#         assert len(psd) == len(self.orig_freqs)
#
#         weights = self.linear_basis_at_linear_nodes_inv \
#             @ np.sum(np.conj(data) / psd * self.linear_basis, axis=1)
#             #* np.exp(-2j * np.pi * time_coalescence * self.orig_freqs), axis=1)
#             #!! time_coalscence didn't work for me... f***ed everything up :( maybe wrong signs? ..
#
#         return weights
#
#     # ? depending on how the inner product is defined in bajes, maybe pull the 4*df out
#     def compute_quadratic_weights(self, psd):
#         """ compte the quadratic weights according to Eq. TODO of my own file
#         """
#         assert len(psd) == len(self.orig_freqs)
#
#         weights = self.quad_basis_at_quad_nodes_inv \
#             @ np.sum(1. / psd * self.quadratic_basis, axis=1)
#
#         return weights


# JenpyROQ: the ROQ data is assumed to be the one provided by https://github.com/gcarullo/JenpyROQ
# Notation and method used here follows `arXiv:1604.08253`.

def compute_linear_weights(tc, args):
    # Equation (9b) in arXiv:1604.08253.
    # Notice the typo: the real part should be taken in Eq.(9a), not in Eq.(9b).
    df, interpolant, data, psd, freqs = args

    vec = np.conj(data)/psd * np.exp(-2.*np.pi*1j*tc*freqs)

    return 4.* df * (interpolant.T @ vec)

def Initialise_pyROQ_Jena_for_inference(path, approx, f_min, f_max, df, freqs_full, psd, data_f, tcs):
    """
        Function to pre-compute the ROQ weights and frequency grid to be used during inference.

        path       : Path of the ROQ data.
        approx     : Waveform approximant for which the ROM was performed.
        f_min      : Minimum frequency at which the model needs evaluation.
        f_max      : Maximum frequency at which the model needs evaluation.
        df         : Frequency step.
        freqs_full : Full array of frequency on which the model would be evaluated in the absence of a ROM.
        psd        : Single-sided power spectral density.
        data_f     : Frequency domain representation of the interferometric data.
    """

    # This try-except is required for compatibility with older PyROQ runs that did not store the metadata. Should be removed in future versions.
    try:
        ROQ_metadata           = {}
        ROQ_metadata_file      = np.genfromtxt(os.path.join(path,'ROQ_data/ROQ_metadata.txt'), names=True)
        ROQ_metadata['f-min']  = ROQ_metadata_file['fmin']
        ROQ_metadata['f-max']  = ROQ_metadata_file['fmax']
        ROQ_metadata['seglen'] = ROQ_metadata_file['seglen']

        # Check that the requested ROQ exists
        # FIXME: still need to check if prior is within training range and that (approx==ROQ_metadata['approx'])
        if not((f_min >= ROQ_metadata['f-min']) and (f_max <=ROQ_metadata['f-max']) and (df>=1./ROQ_metadata['seglen'])): raise ValueError('ROQ data do not exist for the requested parameter estimation settings.')

    except:

        __url__ = 'https://github.com/bernuzzi/PyROQ'
        logger.info("Assuming old PyROQ-refactored format (see {}). Skipping basis compatibility check and cutting the last frequency element.".format(__url__))
        # Old PyROQ basis had one frequency element less.
        data_f, psd, freqs_full = data_f[:-1], psd[:-1], freqs_full[:-1]

    linear_frequencies    = np.load(os.path.join(path,'ROQ_data/linear/empirical_frequencies_linear.npy'))
    quadratic_frequencies = np.load(os.path.join(path,'ROQ_data/quadratic/empirical_frequencies_quadratic.npy'))
    linear_interpolant    = np.load(os.path.join(path,'ROQ_data/linear/basis_interpolant_linear.npy'))
    quadratic_interpolant = np.load(os.path.join(path,'ROQ_data/quadratic/basis_interpolant_quadratic.npy'))

    # FIXME: cut and downsample ROQ interpolants to requested frequency axis, which has to be a subset of the ROQ basis frequency axis.

    logger.info("Starting ROQ linear weights computation, with {} nodes and {} time interpolation points.".format(len(linear_frequencies), len(tcs)))

    # Differently from the quadratic weights, this is a list of weights computed at all the explored times.
    # Each element of the list `omega`, composed by elements `omega_j` is a list over time of the weight computed at `f_j`.
    from .. import eval_func_tuple

    # TODO: do we want to parallelize this step?
    args_linear_weights_tc    = (df, linear_interpolant, data_f, psd, freqs_full)
    linear_weights_tcs_matrix = np.transpose(list(map(eval_func_tuple, zip(repeat(compute_linear_weights), tcs, repeat(args_linear_weights_tc)))))
    linear_weights_interp     = interp1d(tcs, linear_weights_tcs_matrix, kind = 'cubic', bounds_error=False, fill_value=-np.inf)

    logger.info("Starting ROQ quadratic weights computation, with {} frequency nodes.".format(len(quadratic_frequencies)))
    # Equation (10b) in arXiv:1604.08253. Since C is real by construction, no need to take the real part.
    quadratic_weights = 4.* df * quadratic_interpolant.T @ (1.0/psd)

    return linear_frequencies, quadratic_frequencies, linear_weights_interp, quadratic_weights

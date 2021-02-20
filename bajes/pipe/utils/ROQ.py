import numpy as np

def innerProduct(weights, func_1, func_2):#, where=True):
    """ discrete inner L2 Product <f1, f2> = sum_i(weights_i * conjugate(f1)_i * f2_i)
    weights: numpy array
    func_1, func_2: numpy array, (f1, f2)

    return: - inner product of the two (complex number)
    """
    # Experiment using timeit(), from slowest to fastest:
    # - np.sum(a*a, where=True)
    # - np.sum(a*a)
    # - (a*a).sum(where=True)
    # - (a*a).sum()
    return (weights * np.conj(func_1) * func_2).sum()

class ROQ:
    """ Integral Solver using the Reduced Order Quadrature method. The procedure ist taken entirely from:
        Antil, et al.: Two-step greedy algorithm for reduced order quadratures (2012) arXiv:1210.0577v2
    It is used to to calculate the integral of two products of functions <h_(mu_1), h_(mu_2)>.
    Functions h_mu are given by a numpy array

    produces:
    ROQ points, ROQ weights for a given training set
    ROQ points given as actual points where the function is evaluated at ({x_1, x_2, ...}) and the corresponding indices
    """
    def __init__(self, quad_points, quad_weights, training_set_params, func_space, epsilon):
        self.quad_points = np.array(quad_points)
        self.quad_weights = np.array(quad_weights)
        self.training_set_params = training_set_params
        self.epsilon = epsilon

        self.func_space = np.zeros(func_space.shape, dtype=func_space.dtype)
        for i in range(len(self.func_space)):
            self.func_space[i] = self.normalize(func_space[i])
    
    # def createFullROQWeights(self, roq_weights, roq_points, length):
    #     """ create a workable list of ROQ weights (being full length) where it has the correct weight at the ROQ points and its 0 elsewhere

    #     roq_weights: array of roq weights, sorted after roq points (ndarray)
    #     roq_points: indices of the roq points (not sorted) (ndarray dtype=int)
    #     length: full length of ROQ weight array (int)
    #     """
    #     if length < len(roq_points):
    #         raise ValueError("length should be larger or equal to len(roq_points)")

    #     full_roq = np.zeros(length, dtype=roq_weights.dtype)

    #     for i in range(len(roq_points)):
    #         full_roq[roq_points[i]] = roq_weights[i]

    #     return full_roq

    def norm(self, func):
        """ norm of function, using standard weights
        """
        return np.sqrt(np.abs(innerProduct(self.quad_weights, func, func)))
    
    def normalize(self, func):
        """ return copy of func / norm(func), with standard weights for norm
        """
        return func / self.norm(func)
    
    def project(self, projected_func_space, new_basis_element, func_space):
        """ project func onto space spaned by basis (using standard weights for inner product), only adding 
          the new basis element onto projected_func_space

        projected_func_space: iterable consisting of ndarrays, s.t. row denotes which function from func_space and column denotes the value of that function
        new_basis_element: ndarray of new element added to basis
        func_space: space which to project onto space spanned by basis

        len(projected_func_space) (or shape(...)[0]) == len func_space assumed
        """
        #
        # ? TODO: parallize
        for f in range(len(func_space)):
            projected_func_space[f] += innerProduct(self.quad_weights, new_basis_element, func_space[f]) * new_basis_element
        
    def produceROQ(self, output_params=False):
        """ Algorithm 2 from Antil, et al. . Construction of the reduced order quadrature points and weights
          Path #2 chosen, so we wirst approximate F, then F_n² and then F^Tilde (= F_t) (t indicates the tilde)
        
        output_params: if True, the Greedy parameters of the basis for the approximated product step is returned
        
        return: - actual ROQ points (ndarray dtype=Float)
                - ROQ points as list of indices, pointing to the elements in the functions that are considered (evaluated at the ROQ points) (ndarray dtype=int)
                - ROQ weights sorted after ROQ points above (ndarray)
                - [Optional] Greedy Parameters of which elements from the training space make up the final reduced basis (iterable)
        """
        # approximate F
        reduced_basis, greedy_params = self.RBGreedy()

        # don't use reduced_basis but use functions corresponding to greedy parameters, see Remark 2 in Antil, et al.
        greedy_funcs = np.zeros((len(reduced_basis), len(self.func_space[0])), dtype=self.func_space[0].dtype)
        for i in range(len(greedy_params)):
            ind = self.training_set_params.index(greedy_params[i])
            greedy_funcs[i] = self.func_space[ind]

        # approximate F_t (F_n² is approximated inside of twoStepRBGreedy)
        # reduced_basis_t, greedy_params_t = self.twoStepRBGreedy(reduced_basis, greedy_params)
        reduced_basis_t, greedy_params_t = self.twoStepRBGreedy(greedy_funcs, greedy_params)

        # V = [e_1, e_2, ..., e_m] with e_i reduced basis for F_t as columns (M-point arrays)
        V = np.array([e for e in reduced_basis_t]).transpose()

        # DEIM Matrix and points (which serve as the ROQ points)
        DEIM_mat, ROQ_points, ROQ_xval_points = self.DEIM(V)

        # ROQ weights
        # wROQ^T = w^T . V . (P^T . V)^-1 
        ROQ_weights = (self.quad_weights.transpose() @ V @ np.linalg.inv(DEIM_mat.transpose() @ V)).transpose()

        # sort ROQ points and corresponding weights
        sorting_order = np.argsort(ROQ_points)
        ROQ_xval_points = ROQ_xval_points[sorting_order]
        ROQ_points = ROQ_points[sorting_order]
        ROQ_weights = ROQ_weights[sorting_order]

        if output_params:
            return ROQ_xval_points, ROQ_points, ROQ_weights, greedy_params_t
        
        return ROQ_xval_points, ROQ_points, ROQ_weights
    
    def greedyBasis(self, func_space, training_set):
        """ construction of a greedy basis for given function space (as outlined in Algorithm 1 and 4 in Antil, et al.)

        func_space: function space to be approximated (iterable of ndarrays)
        training_set: parameters of func_space (iterable)

        return: - reduced basis of func_space (list of ndarrays)
                - reduced training set (list of iterables)
        """
        n = 0
        sigma = 1
        mu = [ training_set[0] ] # arbitrary choice, training_set does not have to be sorteed
        e = [ func_space[0] ] 
        e[0] = self.normalize(e[0]) # normalize basis elements
        added_elements = [0]    # keep track of all elemnts from the func_space added to the basis so rounding errors do not 
                                # cause the same element to be added multiple times (and indication that the rounding errors added up)
                                # and the desired accuracy couldn't be reached

        projected_funcs = np.zeros((len(func_space), len(func_space[0])), dtype=func_space[0].dtype)

        while sigma >= self.epsilon:
            n += 1
            self.project(projected_funcs, e[n-1], func_space)
            sigmas = np.array([self.norm(func_space[i] - projected_funcs[i]) for i in range(len(func_space))]) # largest error for current subspace e
            index_max = np.argmax(sigmas)
            if index_max in added_elements:
                print("Accuracy couldn't be reached.")
                print("Breaking off at error < " + str(sigmas[index_max]))
                break
            else:
                added_elements.append(index_max)
            sigma = sigmas[index_max]
            print("Error Greedy: " + str(sigma) + ", with basis length: " + str(len(e)))
            mu.append(training_set[index_max])
            e.append(func_space[index_max] - projected_funcs[index_max]) # project onto old basis
            e[n] = self.normalize(e[n]) # normalize element
        
        return e, mu

    def RBGreedy(self):
        """ Algorithm 4 from Antil, et al.

        return: - reduced basis (for F) (list of ndarrays)
                - greedy parameters (for F) (list of iterables)
        """
        return self.greedyBasis(self.func_space, self.training_set_params)
    
    def twoStepRBGreedy(self, reduced_basis, greedy_params):
        """ Algorithm 1 from Antil, et al.

        reduced_basis: basis of which to build the cartesian product (iterable of ndarrays)
        greedy_params: parameters belonging to the reduced basis (list of iterables)

        return: - reduced basis (for F_t) (list of ndarrays)
                - greedy paramters (for F_t) (list of 2-tuples of iterables)
        """
        T2 = [ ]
        for i in range(len(greedy_params)):
            for j in range(len(greedy_params)):
                T2.append((greedy_params[i], greedy_params[j]))
        
        F2 = np.zeros((len(reduced_basis)**2, len(reduced_basis[0])), dtype=reduced_basis[0].dtype)
        for i in range(len(reduced_basis)):
            for j in range(len(reduced_basis)):
                F2[i*j][:] = self.normalize(np.conj(reduced_basis[i]) * reduced_basis[j])
        
        return self.greedyBasis(F2, T2)

    def DEIM(self, V):
        """ Algorithm 5 from Antil, et al.

        V: (Mxm)-matrix as defined in paper. Column vectors fo V must be linearly independent

        return: - interpolation matrix (ndarray shape(Mxm, same as V))
                - interpolation points index positions (ndarray dtype=int)
                - interpolation points (ndarray)
        """
        # already bring them into the correct final shape to avoid costly appending
        j = np.argmax(np.abs(V[:, 0])) # j in [0, M)
        U = np.zeros(np.shape(V), dtype=self.func_space[0].dtype) # Mxm matrix
        P = np.zeros(np.shape(V)) # Mxm matrix
        p_xval = np.zeros(np.shape(V)[1], dtype=self.quad_points.dtype)
        p_ind = np.zeros(np.shape(V)[1], dtype=int) # m vector

        # init U, P, p
        U[:, 0] = V[:, 0] # e_1 as first column
        P[j, 0] = 1 # unit column vector with single unit entry at index j
        p_xval[0] = self.quad_points[0] # actual x values
        p_ind[0] = j # index positions of the x values

        for i in range(1, len(p_ind)): # from 1 to m-1
            c = np.linalg.solve(P[:, :i].transpose() @ U[:, :i], P[:, :i].transpose() @ V[:, i]) # solve (P^T . U) c = (P^T) e_1 for c
            r = V[:, i] - U[:, :i] @ c
            j = np.argmax(np.abs(r))
            U[:, i] = r
            P[j, i] = 1 # we only want unit columns, the rest is already 0
            p_ind[i] = j
            p_xval[i] = self.quad_points[j]
            print("Ready with i=" + str(i) + "/" + str(len(p_ind)))
        
        return P, p_ind, p_xval
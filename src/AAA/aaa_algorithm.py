"""
This script is an adapted version of the "Bayrat" package by Clemens Hofreiter, https://link.springer.com/article/10.1007/s11075-020-01042-0.
Github source: https://github.com/c-f-h/baryrat?tab=readme-ov-file
"""


import numpy as np
import scipy.linalg
import math

try:
    import gmpy2
    import flamp
except ImportError:
    gmpy2 = None
    flamp = None
else:
    from gmpy2 import mpfr, mpc

def _is_mp_array(x):
    """Checks whether `x` is an ndarray containing gmpy2 extended precision numbers."""
    return (gmpy2
            and x.dtype == 'O'
            and len(x) > 0
            and (isinstance(x.flat[0], mpfr) or isinstance(x.flat[0], mpc)))

def _compute_roots(w, x, use_mp):
    # Cf.:
    # Knockaert, L. (2008). A simple and accurate algorithm for barycentric
    # rational interpolation. IEEE Signal processing letters, 15, 154-157.
    #
    # This version requires solving only a standard eigenvalue problem, but
    # has troubles when the polynomial has leading 0 coefficients.
    if _is_mp_array(w) or _is_mp_array(x):
        use_mp = True

    if use_mp:
        assert flamp, 'flamp package is not installed'
        ak = flamp.to_mp(w)     # TODO: this always copies!
        bk = flamp.to_mp(x)
        ak /= sum(ak)
        M = np.diag(bk) - np.outer(ak, x)
        lam = flamp.eig(M, left=False, right=False)
        # remove one simple root
        lam = np.delete(lam, np.argmin(abs(lam)))
        return lam
    else:
        # the same procedure in standard double precision
        ak = w / w.sum()
        M = np.diag(x) - np.outer(ak, x)
        lam = scipy.linalg.eigvals(M)
        # remove one simple root
        lam = np.delete(lam, np.argmin(abs(lam)))
        return np.real_if_close(lam)

def _compute_roots2(z, f, w):
    # computation of roots/poles by companion matrix pair; see, e.g.:
    #   Fast Reduction of Generalized Companion Matrix Pairs for
    #   Barycentric Lagrange Interpolants,
    #   Piers W. Lawrence, SIAM J. Matrix Anal. Appl., 2013
    #   https://doi.org/10.1137/130904508
    #
    # This version can deal with leading 0 coefficients of the polynomial, but
    # requires solving a generalized eigenvalue problem, which is currently not
    # supported in mpmath/flamp.
    B = np.eye(len(w) + 1)
    B[0,0] = 0
    E = np.block([[0, w],
                  [f[:,None], np.diag(z)]])
    evals = scipy.linalg.eigvals(E, B)
    return np.real_if_close(evals[np.isfinite(evals)])


class BarycentricRational:
    """A class representing a rational function in barycentric representation.

    Args:
        z (array): the interpolation nodes
        f (array): the values at the interpolation nodes
        w (array): the weights

    The rational function has the interpolation property r(z_j) = f_j at all
    nodes where w_j != 0.
    """
    def __init__(self, z, f, w):
        if not (len(z) == len(f) == len(w)):
            raise ValueError('arrays z, f, and w must have the same length')
        self.nodes = np.asanyarray(z)
        self.values = np.asanyarray(f)
        self.weights = np.asanyarray(w)

    def __call__(self, x):
        """Evaluate rational function at all points of `x`."""
        zj,fj,wj = self.nodes, self.values, self.weights

        xv = np.asanyarray(x).ravel()
        if len(xv) == 0:
            return np.empty(np.shape(x), dtype=xv.dtype)
        D = xv[:,None] - zj[None,:]
        # find indices where x is exactly on a node
        (node_xi, node_zi) = np.nonzero(D == 0)

        one = xv[0] * 0 + 1     # for proper dtype when using MP

        with np.errstate(divide='ignore', invalid='ignore'):
            if len(node_xi) == 0:       # no zero divisors
                C = np.divide(one, D)
                r = C.dot(wj * fj) / C.dot(wj)
            else:
                # set divisor to 1 to avoid division by zero
                D[node_xi, node_zi] = one
                C = np.divide(one, D)
                r = C.dot(wj * fj) / C.dot(wj)
                # fix evaluation at nodes to corresponding fj
                # TODO: this is only correct if wj != 0
                r[node_xi] = fj[node_zi]

        if np.isscalar(x):
            return r[0]
        else:
            r.shape = np.shape(x)
            return r

    def uses_mp(self):
        """Checks whether any of the data of this rational function uses
        extended precision.
        """
        return _is_mp_array(self.nodes) or _is_mp_array(self.values) or _is_mp_array(self.weights)


    @property
    def order(self):
        """The order of the barycentric rational function, that is, the maximum
        degree that its numerator and denominator may have, or the number of
        interpolation nodes minus one.
        """
        return len(self.nodes) - 1

    def poles(self, use_mp=False):
        """Return the poles of the rational function.

        If ``use_mp`` is ``True``, uses the ``flamp`` multiple precision
        package to compute the result. This option is automatically enabled if
        :meth:`uses_mp` is True.
        """
        if use_mp or self.uses_mp():
            return _compute_roots(self.weights, self.nodes, use_mp=True)
        else:
            return _compute_roots2(self.nodes, np.ones_like(self.values), self.weights)

    def polres(self, use_mp=False):
        """Return the poles and residues of the rational function.

        If ``use_mp`` is ``True``, uses the ``flamp`` multiple precision
        package to compute the result. This option is automatically enabled if
        :meth:`uses_mp` is True.
        """
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        if self.uses_mp():
            use_mp = True

        # compute poles
        pol = self.poles(use_mp=use_mp)


        # compute residues via formula for simple poles of quotients of analytic functions
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            C_pol = 1.0 / (pol[:,None] - zj[None,:])
            N_pol = C_pol.dot(fj*wj)
            Ddiff_pol = (-C_pol**2).dot(wj)
            res = N_pol / Ddiff_pol

            res = np.where(np.isnan(res), 0, res)#remove nans that are caused by division by zero in C_pol

        return pol, res

    def zeros(self, use_mp=False):
        """Return the zeros of the rational function.

        If ``use_mp`` is ``True``, uses the ``flamp`` multiple precision
        package to compute the result. This option is automatically enabled if
        :meth:`uses_mp` is True.
        """
        if use_mp or self.uses_mp():
            return _compute_roots(self.weights*self.values, self.nodes,
                    use_mp=True)
        else:
            return _compute_roots2(self.nodes, self.values, self.weights)

    def gain(self):
        """The gain in a poles-zeros-gain representation of the rational function,
        or equivalently, the value at infinity.
        """
        return np.sum(self.values * self.weights) / np.sum(self.weights)


    def numerator(self):
        """Return a new :class:`BarycentricRational` which represents the numerator polynomial."""
        weights = _polynomial_weights(self.nodes)
        return BarycentricRational(self.nodes.copy(), self.values * self.weights / weights, weights)

    def denominator(self):
        """Return a new :class:`BarycentricRational` which represents the denominator polynomial."""
        weights = _polynomial_weights(self.nodes)
        return BarycentricRational(self.nodes.copy(), self.weights / weights, weights)

    def degree_numer(self, tol=1e-12):
        """Compute the true degree of the numerator polynomial.

        Uses a result from [Berrut, Mittelmann 1997].
        """
        N = len(self.nodes) - 1
        for defect in range(N):
            if abs(np.sum(self.values * self.weights * (self.nodes ** defect))) > tol:
                return N - defect
        return 0

    def degree_denom(self, tol=1e-12):
        """Compute the true degree of the denominator polynomial.

        Uses a result from [Berrut, Mittelmann 1997].
        """
        N = len(self.nodes) - 1
        for defect in range(N):
            if abs(np.sum(self.weights * (self.nodes ** defect))) > tol:
                return N - defect
        return 0

    def degree(self, tol=1e-12):
        """Compute the pair `(m,n)` of true degrees of the numerator and denominator."""
        return (self.degree_numer(tol=tol), self.degree_denom(tol=tol))

    
################################################################################

def aaa(Z, F, tol=1e-13, mmax=100, return_errors=False):
    """Compute a rational approximation of `F` over the points `Z` using the
    AAA algorithm.

    Arguments:
        Z (array): the sampling points of the function. Unlike for interpolation
            algorithms, where a small number of nodes is preferred, since the
            AAA algorithm chooses its support points adaptively, it is better
            to provide a finer mesh over the support.
        F: the function to be approximated; can be given as a function or as an
            array of function values over ``Z``.
        tol: the approximation tolerance
        mmax: the maximum number of iterations/degree of the resulting approximant
        return_errors: if `True`, also return the history of the errors over
            all iterations

    Returns:
        BarycentricRational: an object which can be called to evaluate the
        rational function, and can be queried for the poles, residues, and
        zeros of the function.

    For more information, see the paper

      | The AAA Algorithm for Rational Approximation
      | Yuji Nakatsukasa, Olivier Sete, and Lloyd N. Trefethen
      | SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522
      | https://doi.org/10.1137/16M1106122

    as well as the Chebfun package <http://www.chebfun.org>. This code is an
    almost direct port of the Chebfun implementation of aaa to Python.
    """
    Z = np.asanyarray(Z).ravel()
    if callable(F):
        # allow functions to be passed
        F = F(Z)
    F = np.asanyarray(F).ravel()

    J = list(range(len(F)))
    zj = np.empty(0, dtype=Z.dtype)
    fj = np.empty(0, dtype=F.dtype)
    C = []
    errors = []

    reltol = tol * np.linalg.norm(F, np.inf)

    R = np.mean(F) * np.ones_like(F)

    for _ in range(mmax):
        # find largest residual
        jj = np.argmax(abs(F - R))
        zj = np.append(zj, (Z[jj],))
        fj = np.append(fj, (F[jj],))
        J.remove(jj)

        # Cauchy matrix containing the basis functions as columns
        C = 1.0 / (Z[J,None] - zj[None,:])
        # Loewner matrix
        A = (F[J,None] - fj[None,:]) * C

        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        wj = Vh[-1, :].conj()

        # approximation: numerator / denominator
        N = C.dot(wj * fj)
        D = C.dot(wj)

        # update residual
        R = F.copy()
        R[J] = N / D

        # check for convergence
        errors.append(np.linalg.norm(F - R, np.inf))
        if errors[-1] <= reltol:
            break

    r = BarycentricRational(zj, fj, wj)
   
    return (r, errors) if return_errors else r


def cleanup(r, Z, F):
    """
    Function to remove the Froissart doublets, i.e. selected points with vansihing residues.
    Algorithm is described in: https://doi.org/10.1137/16M110612

    Parameters:
    r: instance of BarycentricRational
    Z: Original "fine grid"
    F: Function values on original "fine grid"

    Returns: 
    r : cleaned BarycentricRational object, i.e. without Froissart doublets

    """
    zj, fj, wj = r.nodes, r.values, r.weights
    Z = Z.copy()
    F = F.copy()

    #compute poles and residues
    poles, residues = r.polres()

    ii = np.where(np.abs(residues) < 1e-13)[0] #indices with vanishing residues (Froissart doublets)
    ni = len(ii) # number of Froissart doublets 
    if ni == 0:
        return r

    print(f'{ni} Froissart doublets')

    for j in range(ni):
        azp = np.abs(zj - poles[ii[j]]) # distance of nodes to the "Froissart-pole" with index j
        jj = np.argmin(azp) # index of the node closest to the pole
        zj = np.delete(zj, jj) # remove the node from the list of nodes
        fj = np.delete(fj, jj) # remove the corresponding function value

    for j in range(len(zj)): #loop through updates nodes in order to determine the set of "unselected" grid points
        idx = np.where(Z == zj[j])[0] # find the index of the node in the original "fine" grid
        Z = np.delete(Z, idx) # remove the node from the original "fine" grid
        F = np.delete(F, idx) # remove the node from the function values on the "fine grid"

        #at this stage, Z and F correspond to the unselected points, i.e. those not contained in z and f.

    # Cauchy matrix containing the basis functions as columns
    C = 1.0 / (Z[:,None] - zj[None,:])
    # Loewner matrix
    A = (F[:,None] - fj[None,:]) * C

    # compute weights as right singular vector for smallest singular value
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    wj = Vh[-1, :].conj()

    #update r to "cleaned" object
    r = BarycentricRational(zj, fj, wj)
    return r



def _polynomial_weights(x):
    n = len(x)
    w = np.array([
            1.0 / np.prod([x[i] - x[j] for j in range(n) if j != i])
            for i in range(n)
    ])
    return w / np.abs(w).max()
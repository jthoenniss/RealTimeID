import numpy as np
import DiscrError as de
import scipy.linalg.interpolative as sli
from utils import common_funcs as cf



class RtKernel:  # class that compute the ID and SVD for given parameters.
    DEFAULT_PHI = np.pi / 4

    def __init__(self, m, n, beta, times, eps, h, phi=DEFAULT_PHI):
        """
        Parameters:
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - beta (float): inverse temperature
        - times (numpy.ndarray): Array containing the points on the time grid
        - eps (float): error for interpolvative decomposition (ID) and singular value decomposition (SVD)
        - h (float): Discretization parameter
        - phi (float): rotation angle in complex plane
        """

        self.m, self.n = m, n
        self.beta = beta
        self.times = times
        self.eps = eps
        self.h = h
        self.phi = phi

        # initialize frequency grid
        self.fine_grid = np.array(
            [
                np.exp(self.h * k - np.exp(-self.h * k))
                for k in range(-self.n, self.m + 1)
            ]
        )  # create fine grid according to superexponential formula

        # create kernel matrix K on fine grid
        self.K = np.array(
            [
                [
                    cf.distr(
                        t,
                        np.exp(self.h * k - np.exp(-self.h * k))
                        * np.exp(1.0j * self.phi),
                        beta,
                    )
                    * self.h
                    * np.exp(1.0j * self.phi)
                    * (1 + np.exp(-self.h * k))
                    * np.exp(self.h * k - np.exp(-self.h * k))
                    for k in range(-self.n, self.m + 1)
                ]
                for t in self.times
            ]
        )

        # __________perform SVD on kernel K and count number of singular values above error threshold____________
        (
            self.num_singular_values_above_threshold,
            self.singular_values,
        ) = cf.svd_check_singular_values(self.K, self.eps)

        # Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank (see documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html" (access: 6. Dec. 2023))
        assert (
            self.eps < 1
        ), "'eps' needs to be smaller than 1 to be interpreted as an error and not as a rank (see scipy documentation for ID)"
        # perform ID on K
        self.ID_rank, self.idx, self.proj = sli.interp_decomp(
            self.K, self.eps
        )  # Comment: The fast version of this algorithm from the scipy library uses random sampling and may not give completely identical results for every run. See documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html". Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank, see documentation (access: 6. Dec. 2023)

        # compute coarse grid
        self.coarse_grid = np.array(self.fine_grid[self.idx[: self.ID_rank]])


class RtDlr(RtKernel):
    def __init__(self, params):
        """
        Params (passed as dictionary or implicitly as part of a DiscrError object):
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - beta (float): inverse temperature
        - times (numpy.ndarray): Array containing the points on the time grid
        - eps (float): error used for interpolvative decomposition (ID) and singular value decomposition (SVD). When initialized through DiscrError object, this error is the rel. error between the discrete and continous-frequency integral
        - h (float): Discretization parameter
        - phi (float): rotation angle in complex plane
        """
        members_DiscrError = ["m", "n", "beta", "times", "eps", "h", "phi"]

        #for intitialization via dict containing these variables
        if isinstance(params, dict):
            #Extract values from dictionary
            m, n, beta, times, eps, h, phi = (
                params.get(member) for member in members_DiscrError
            )

        #for initialization using DiscrError object
        elif isinstance(params, de.DiscrError):
            # Extract values from the DiscrError object
            m, n, beta, times, eps, h, phi = (
                getattr(params, member) for member in members_DiscrError
            )

        else:
            raise ValueError(
                "Invalid type for params. Should be either a dictionary or an instance of class DiscrError."
            )

        # initialize object of parent class RtKernel
        super().__init__(m, n, beta, times, eps, h, phi)
        # ____________________________________________________

    def get_coarse_grid(self):
        """
        Return coarse frequency grid containing the frequencies chosen by the interpolative decomposition
        """
        return self.coarse_grid

    def spec_dens_fine(self, phi=None):
        """
        Evaluate the spectral density at the frequency points of the fine grid
        Parameters:
        - phi (float): rotation angle in complex plane

        Returns:
        - np.complex: spectral density evaluated at complex frequencys of fine grid (rotated into complex plane)
        """
        if phi is None:
            phi_cmplx = self.phi
        else:
            phi_cmplx = phi

        spec_dens_at_fine_grid = np.array(
            [cf.spec_dens(w_f * np.exp(1.0j * phi_cmplx)) for w_f in self.fine_grid]
        )
        return spec_dens_at_fine_grid

    def get_projection_matrix(self):
        """
        Compute the projection matrix needed to compute effective couplings
        """
        P = np.hstack([np.eye(self.ID_rank), self.proj])[
            :, np.argsort(self.idx)
        ]  # projection matrix

        return P

    def coupl_eff(self):
        """
        Compute effective couplings: multiply vector of spectral density at fine grid points with projection matrix P.
        """
        P = self.get_projection_matrix()
        coupl_eff = P @ self.spec_dens_fine()
        return coupl_eff


    def reconstr_interp_matrix(self):
        """
        Parameters:

        Returns:
        2D matrix with np.complex_: ID reconstructed matrix
        """

        # _____reconstruct interpolation matrix_________
        B = sli.reconstruct_skel_matrix(self.K, self.ID_rank, self.idx)
        # reconstructed interpolation matrix:
        K_reconstr = sli.reconstruct_matrix_from_id(B, self.idx, self.proj)

        return K_reconstr

    def reconstruct_propag(self, compute_error=False):
        """
        Reconstruct the propagator with ID approximation. Optionally also compute error to original propagator
        Parameters:
        compute_error (bool): flag that decides wheather the relative time-integrated error to original propagator is computed

        Returns:
        - np.array(np.complex_): propagator at all time points of fine grid
        - float: relative error between original and reconstructed Green's function [only if compute_error flag is set to 'True']
        """
        K_reconstr = self.reconstr_interp_matrix()  # ID-reconstructed kernel matrix
        Gamma = self.spec_dens_fine()  # spectral denstiy evaluated on fine grid points

        G_reconstr = (
            K_reconstr @ Gamma
        )  # this yields the propagators where the array elements correspond to the different time points

        yield G_reconstr  # return reconstructed Green's function

        if compute_error:  # evaluate error if flag is true
            G_orig = self.K @ Gamma

            error_rel = np.sum(abs(G_orig - G_reconstr)) / np.sum(
                abs(G_orig) + abs(G_reconstr)
            )  # in the relative error, the time steps cancels out and is thus not needed here.

            yield error_rel

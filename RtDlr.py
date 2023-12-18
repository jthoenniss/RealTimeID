"""
This module defines the `RtKernel` class and its inherited class `RtDlr`.

### RtKernel:
The `RtKernel` class is responsible for creating the Green's function kernel, 
incorporating real-time arguments and fermionic statistics, including 
temperature-dependent beta. Its primary function is to perform the Interpolative Decomposition (ID) 
and Singular Value Decomposition (SVD), storing the relevant data.

### RtDlr:
The `RtDlr` class encompasses various initialization routines and functions 
that compute quantities based on the ID decomposition. It facilitates the analysis of the effective, 
ID-reconstructed kernel, Green's function, and more. This class serves as an extension of the `RtKernel` class, 
inheriting its functionality.
"""


import numpy as np
import DiscrError as de
import scipy.linalg.interpolative as sli
from utils import common_funcs as cf


class RtKernel:  # class that compute the ID and SVD for given parameters.
    DEFAULT_PHI = np.pi / 4
    MAX_EPS = 1.0

    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        times: np.ndarray,
        eps: float,
        h: float,
        phi: float = DEFAULT_PHI,
    ):
        r"""
        Initialize the RtKernel class with given parameters.

        The kernel is calculated using the formula:

        $$ K(t, \omega) = \exp(i \omega t) \frac{1}{1 + \exp(-\beta \omega)} $$

        where:
        - $\omega$ is the frequency,
        - $t$ is the time,
        - $\beta$ is the inverse temperature (1/T) [$k_B = 1$].

        Parameters:
        - m (int): Number of discretization intervals for $\omega > 1$
        - n (int): Number of discretization intervals for $\omega < 1$
        - beta (float): Inverse temperature ($\beta$)
        - times (numpy.ndarray): Array containing the points on the time grid
        - eps (float): Error for interpolative decomposition (ID) and singular value decomposition (SVD)
        - h (float): Discretization parameter
        - phi (float): Rotation angle in the complex plane
        """

        self.validate_parameters(eps)  # check that error parameter is valid
        self.initialize_parameters(m, n, beta, times, eps, h, phi)
        self.fine_grid, self.K = self.create_kernel()

        # __________perform SVD on kernel K and count number of singular values above error threshold____________
        (
            self.num_singular_values_above_threshold,
            self.singular_values,
        ) = self.perform_svd()

        # ______perform ID on kernel K
        self.ID_rank, self.idx, self.proj = self.perform_ID()
        #compute coarse ID grid
        self.coarse_grid = self.compute_coarse_grid()

    def validate_parameters(self, eps: float):
        # Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank (see documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html" (access: 6. Dec. 2023))
        if not 0 < eps < RtKernel.MAX_EPS:
            raise ValueError(
                f"'eps' must be between 0 and {RtKernel.MAX_EPS}, got {eps}"
            )

    def initialize_parameters(
        self, m: int, n: int, beta: float, times: np.ndarray, eps: float, h: float, phi
    ):
        self.m, self.n = m, n
        self.beta = beta
        self.times = times
        self.eps = eps
        self.h = h
        self.phi = phi

    def create_kernel(self):
        """
        Create the kernel matrix on the fine grid in the complex plane.

        This method uses class attributes such as the rotation angle (`phi`),
        discretization parameter (`h`), inverse temperature (`beta`), and
        the frequency (`m`, `n`) and time (`times`) grids to generate the kernel matrix.
        """
        k_values = np.arange(-self.n, self.m + 1)
        t_grid = self.times[:, np.newaxis]  # reshape for broadcasting
        fine_grid, K = self.compute_kernel_matrix(k_values, t_grid)
        return fine_grid, K

    def compute_kernel_matrix(self, k_values, t_grid):
        fine_grid = np.exp(self.h * k_values - np.exp(-self.h * k_values))
        fine_grid_complex = fine_grid * np.exp(1.0j * self.phi)
        K = cf.distr(t_grid, fine_grid_complex, self.beta)
        K *= self.h * (1 + np.exp(-self.h * k_values)) * fine_grid_complex
        return fine_grid, K

    def perform_svd(self):
        (
            num_singular_values_above_threshold,
            singular_values,
        ) = cf.svd_check_singular_values(self.K, self.eps)
        return num_singular_values_above_threshold, singular_values

    def perform_ID(self):
        # perform ID on K
        ID_rank, idx, proj = sli.interp_decomp(
            self.K, self.eps
        )  # Comment: The fast version of this algorithm from the scipy library uses random sampling and may not give completely identical results for every run. See documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html". Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank, see documentation (access: 6. Dec. 2023)
        return ID_rank, idx, proj

    def compute_coarse_grid(self):
        # compute coarse grid
        coarse_grid = np.array(self.fine_grid[self.idx[: self.ID_rank]])

        return coarse_grid


class RtDlr(RtKernel):
    def __init__(self, *args, **kwargs):
        """
        Parameters are passed as
        a) dictionary: "dlr.RtDlr(dict)",
        b) implicitly as part of a DiscrError object: "dlr.RtDlr(discr_error)",
        c) as as keyword arguments: "dlr.RtDlr(m=10, n=20, beta=10., times=[1, 2], eps=0.3, h=0.2)".

        Necessary arguments are:
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - beta (float): inverse temperature
        - times (numpy.ndarray): Array containing the points on the time grid
        - eps (float): error used for interpolative decomposition (ID) and singular value decomposition (SVD).
          When initialized through DiscrError object, this error is the rel. error between the discrete
          and continuous-frequency integral
        - h (float): Discretization parameter
        - phi (float): rotation angle in the complex plane
        """
        if args:
            self._initialize_from_args(args)
        elif kwargs:
            super().__init__(**kwargs)
        else:
            raise ValueError(
                "Invalid parameters. Provide either a dictionary or an instance of DiscrError."
            )

    # Function definitions used in initilization:
    def _initialize_from_args(self, args):
        if isinstance(args[0], de.DiscrError):
            self._initialize_from_discr_error(args[0])
        elif isinstance(args[0], dict):
            super().__init__(**args[0])
        elif args[0] is None:
            self._initialize_from_none()
        else:
            raise ValueError(
                "Invalid type for args[0]. Should be either a dictionary or an instance of DiscrError."
            )

    def _initialize_from_discr_error(self, discr_error):
        params_rt_kernel = ["m", "n", "beta", "times", "eps", "h", "phi"]
        super().__init__(
            **{member: getattr(discr_error, member) for member in params_rt_kernel}
        )

    def _initialize_from_none(self):
        # Do not call the base class initializer. Set all associated attributes to trivial values.

        # Integer attributes
        integer_attributes = [
            "m",
            "n",
            "num_singular_values_above_threshold",
            "ID_rank",
        ]
        for member in integer_attributes:
            setattr(self, member, 0)

        # Float attributes
        float_attributes = ["beta", "eps", "h", "phi", "singular_values"]
        for member in float_attributes:
            setattr(self, member, np.NaN)

        # Array attributes
        array_attributes = ["times", "fine_grid", "K", "idx", "proj", "coarse_grid"]
        for member in array_attributes:
            setattr(self, member, np.array([]))

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

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
from src.discr_error import DiscrError as de
import scipy.linalg.interpolative as sli
from src.utils import common_funcs as cf
from src.rt_kernel.parameter_validator import ParameterValidator


class RtKernel:  # class that compute the ID and SVD for given parameters.
    DEFAULT_PHI = np.pi / 4
    MAX_EPS = 1.0

    # _________Initialize class_______
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
        # initialize all parameters (implicitly checked for validity in initialize_parameters)
        self.initialize_parameters(m, n, beta, times, eps, h, phi)

        # compute the fine grid and the kernel matrix K
        self.fine_grid, self.K = self.create_kernel()

        # Perform SVD on kernel K and count number of singular values above error threshold____________
        (
            self.num_singular_values_above_threshold,
            self.singular_values,
        ) = self.perform_svd()

        # Perform ID on kernel K
        self.ID_rank, self.idx, self.proj = self.perform_ID()
        # compute coarse ID grid
        self.coarse_grid = self.compute_coarse_grid()

    # _______________End Initialization Routine___________________________

    def initialize_parameters(
        self,
        m: int,
        n: int,
        beta: float,
        times: np.ndarray,
        eps: float,
        h: float,
        phi: float,
    ):
        """
        Initializes class parameters after validation and stores them as attributes.
        """
        # check if all parameters are valid
        ParameterValidator.validate_m_n(m, n)
        ParameterValidator.validate_beta(beta)
        ParameterValidator.validate_times(times)
        ParameterValidator.validate_eps(eps)
        ParameterValidator.validate_h(h)
        ParameterValidator.validate_phi(phi)

        # Store attributes
        self.m, self.n = m, n
        self.beta = beta
        self.times = times
        self.eps = eps
        self.h = h
        self.phi = phi

    def create_kernel(self):
        """
        Create the kernel matrix on the fine grid in the complex plane.

        This method computes a kernel matrix using class attributes, incorporating
        a complex rotation specified by `phi`. The kernel is calculated over
        a range of frequency values (defined by `m` and `n`) and time values
        (defined by `times`).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the fine grid array
            and the computed kernel matrix.
        """

        k_values = np.arange(-self.n, self.m + 1)
        t_grid = self.times[:, np.newaxis]  # reshape for broadcasting
        fine_grid, K = self._compute_kernel_matrix(t_grid, k_values)
        return fine_grid, K

    def _compute_kernel_matrix(self, t_grid, k_values):
        r"""
        Computes the kernel matrix using 'k_valus', which defines the exponentially discretized frequency grid, the time grid, and class attributes.

        Parameters:
            k_values (np.ndarray): Integer values defining the exponential frequency grid that is defined via $\omega_k = \exp{(h k - \exp{- h k})}.$
            t_grid (np.ndarray): Time grid for kernel computation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The fine grid and computed kernel matrix.
        """
        exp_k_values = np.exp(
            -self.h * k_values
        )  # precompute as it is needed repeatedly
        fine_grid = np.exp(
            self.h * k_values - exp_k_values
        )  # exponential frequency grid with as many points as 'k_values' has entries
        fine_grid_complex = fine_grid * np.exp(
            1.0j * self.phi
        )  # fine grid rotated into complex plane by angle phi.
        K = cf.distr(
            t_grid, fine_grid_complex, self.beta
        )  # kernel defined by Fermi-distribution cf.distr and the time-dependent Fourier factor e^{i\phi t}
        K *= (
            self.h * (1 + exp_k_values) * fine_grid_complex
        )  # add factors stemming from the variable transformation from \omega to k (multiply rows of K elementwise).
        return fine_grid, K

    def perform_svd(self):
        """
        Performs SVD on `self.K`, returning the count of singular values above `self.eps` and the values themselves.

        Returns:
            Tuple[int, np.ndarray]: Count of singular values above threshold and array of singular values.
        """
        (
            num_singular_values_above_threshold,
            singular_values,
        ) = cf.svd_check_singular_values(self.K, self.eps)
        return num_singular_values_above_threshold, singular_values

    def perform_ID(self):
        """
        Performs interpolative decomposition (ID) on `self.K` using `self.eps` as the error threshold.
        # Comment: The fast version of this algorithm from the scipy library uses random sampling and may not give completely identical results for every run. See documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html". Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank, see documentation (access: 6. Dec. 2023)

        Returns:
            Tuple[int, np.ndarray, np.ndarray]: The rank of ID, indices, and projection matrix.
        """
        ID_rank, idx, proj = sli.interp_decomp(self.K, self.eps)
        return ID_rank, idx, proj

    def compute_coarse_grid(self):
        """
        Compute coarse grid which consists of the frequencies selected by the ID from the fine grid
        """
        coarse_grid = np.array(self.fine_grid[self.idx[: self.ID_rank]])

        return coarse_grid


class RtDlr(RtKernel):
    # Define required parameters as a class variable
    REQUIRED_PARAMS = ["m", "n", "beta", "times", "eps", "h", "phi"]

    def __init__(self, *args, **kwargs):
        """
        Parameters are passed as
        a) dictionary: "dlr.RtDlr(dict)",
        b) implicitly as attributes of an object, e.g. of type DiscrError: "dlr.RtDlr(discr_error)",
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
            self._validate_and_initialize(kwargs)
        else:
            self._initialize_with_defaults()

    # Function definitions used in initilization:
    def _initialize_from_args(self, args):
        """
        Initializes the object from the first argument in 'args'.

        The method supports initialization in three ways:
        1. From a dictionary: Interprets the first argument as parameter key-value pairs.
        2. From an object: Uses attributes of the first argument if it's not a dictionary.
        3. With defaults: Initializes with default values if the first argument is None.
        """
        if isinstance(args[0], dict):
            self._validate_and_initialize(**args[0])
        elif args[0] is None:
            self._initialize_with_defaults()
        else:
            self._initialize_from_object(args[0])

    def _initialize_from_object(self, obj):
        """
        Initialize the class from an object 'obj' (e.g. of type DiscrError) from which we extract the attributes.
        """
        init_params = {
            param: getattr(obj, param)
            for param in RtDlr.REQUIRED_PARAMS
            if hasattr(obj, param)
        }  # check for each required attribute in RtDlr.REQUIRED_PARAMS
        self._validate_and_initialize(
            init_params
        )  # check if all required attributes are there, if yes, initialize, otherwise error will be thrown

    def _validate_and_initialize(self, params):
        """
        Validate parameters (i.e. check if all required arguments are present) and initialize the class.
        """
        ParameterValidator.validate_required_params(
            params, RtDlr.REQUIRED_PARAMS
        )  # check if all required attributes are there,
        super().__init__(
            **params
        )  # if the previous step didn't raise an exception, initialize

    def _initialize_with_defaults(self):
        """
        Private method to initialize with default values when no arguments are provided or argument is None.
        In this case, the initializer of the parent class Rt_Kernel is not called.
        """
        # Default initialization of integer, float, and array attributes
        integer_attributes = [
            "m",
            "n",
            "num_singular_values_above_threshold",
            "ID_rank",
        ]
        float_attributes = ["beta", "eps", "h", "phi", "singular_values"]
        array_attributes = ["times", "fine_grid", "K", "idx", "proj", "coarse_grid"]

        for member in integer_attributes:
            setattr(self, member, 0)
        for member in float_attributes:
            setattr(self, member, 0.0)
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
        - numpy.ndarray: Spectral density evaluated at complex frequencies of the fine grid (rotated into the complex plane).
        """
        phi_cmplx = self.phi if phi is None else phi

        rotated_frequencies = self.fine_grid * np.exp(1.0j * phi_cmplx)
        spec_dens_at_fine_grid = cf.spec_dens(rotated_frequencies)

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

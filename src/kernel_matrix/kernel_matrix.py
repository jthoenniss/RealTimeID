import numpy as np
from src.utils import common_funcs as cf
from src.kernel_params.kernel_params import KernelParams
from typing import Tuple


class KernelMatrix:

    """
    A class that generates the kernel matrix associated with a Green's function.

    This class contains functions to compute a fine frequency grid, initialize the kernel matrix,
    and update the kernel matrix based on parameter changes.

    Parameters:
        - m (int): Number of discretization intervals for omega > 1/e.
        - n (int): Number of discretization intervals for omega < 1/e.
        - beta (float): Inverse temperature.
        - N_max (int): Number of points on the time grid.
        - delta_t (float): Time step.
        - h (float): Discretization parameter.
        - phi (float): Rotation angle in the complex plane.
        - spec_dens (callable): Spectral density as a function with one parameter
        - freq_parametrization (str): The parameterization of the frequency grid. Options are "simple_exp" and "fancy_exp".
            Simple exp: The grid is parametrized by omega_k = exp(h*k) for k in [-n, m].
            Fancy exp: The grid is parametrized by omega_k = exp(h*k - exp(-h*k)) for k in [-n, m].
    """

    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        N_max: int,
        delta_t: float,
        h: float,
        phi: float,
        spec_dens: callable,
        freq_parametrization: str,
    ):
        # check if all parameters are valid
        KernelParams.validate_m_n(m, n)
        KernelParams.validate_beta(beta)
        KernelParams.validate_N_max_and_delta_t(N_max, delta_t)
        KernelParams.validate_h(h)
        KernelParams.validate_phi(phi)
        KernelParams.validate_freq_parametrization(freq_parametrization)

        # Store parameters
        self.m, self.n = m, n
        self.beta = beta
        self.N_max = N_max
        self.delta_t = delta_t
        self.h = h
        self.phi = phi
        self.spec_dens = spec_dens
        self.freq_parametrization = freq_parametrization

        # Initialize kernel matrix and grids
        self._initialize_kernel_and_grids()

    def _initialize_kernel_and_grids(
        self,
    ) -> None:
        """
        (Re)compute the time grid, fine frequency grid, the kernel matrix and the vectorized spectrald density.
        Needed for initialization and after change of parameters.
        Parameters:
        - None
        Returns:
        - None
        """
        # set time grid
        self.times = cf.set_time_grid(N_max=self.N_max, delta_t=self.delta_t)
        # initialize frequency grid
        self.fine_grid, self.k_values, jacobian = self._initialize_fine_grid()
        # initialize matrix kernel
        self.kernel = self._initialize_kernel(jacobian=jacobian)
        # initialize the spectral density as 1 for all values of the fine grid (spec_dens is included in the kernel matrix)
        self.spec_dens_array_fine = np.ones_like(self.fine_grid)


    def _initialize_fine_grid(self) -> tuple:
        """
        Generates a fine grid for given discretization parameters.
        Parameters:
        - None
        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - The generated fine grid as a NumPy array.
        - The k values that define the grid points.
        - The Jacobian of the transformation from the measure (dw/dk).
        """
        k_values = np.arange(-self.n, self.m + 1)

        #initialize fine grid and Jacobian with exact implementation depending on the grid parametrization
        if self.freq_parametrization == "simple_exp":
            fine_grid = np.exp(self.h * k_values)
            jacobian = self.h * fine_grid

        elif self.freq_parametrization == "fancy_exp":
            fine_grid = np.exp(self.h * k_values - np.exp(-self.h * k_values))
            jacobian = self.h * (1 + np.exp(-self.h * k_values)) * fine_grid  # Jacobian from measure. This is dw/dk.
       
        else:
            raise ValueError("Invalid grid parametrization argument. Must be 'simple_exp' or 'fancy_exp'. Got: " + self.freq_parametrization)
        
        return fine_grid, k_values, jacobian

    def _initialize_kernel(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Creates the kernel matrix using the Fermi distribution function and spectral density.
        Parameters:
        - jacobian (np.ndarray): The Jacobian of the transformation from the measure: dw/dk.

        Returns:
        - np.ndarray: Kernel matrix.
        """
        times_arr = self.times[:, np.newaxis]  # enable broadcasting
        fine_grid_complex = self.fine_grid * np.exp(1.0j * self.phi)
        spec_dens_array_cmplx = self._compute_spec_dens_array_cmplx()

        # Kernel defined by Fermi distribution and spectral density
        K = cf.distr(times_arr, fine_grid_complex, self.beta) * spec_dens_array_cmplx
        K *= jacobian * np.exp(1.j * self.phi) # multiply by Jacobian from measure. This is dw/dk. Multiply by exp(i*phi) to rotate in the complex plane

        return K

    def _compute_spec_dens_array_cmplx(self) -> np.ndarray:
        """
        Helper function for vecotized computation

        Returns:
        np.ndarray: array with values of spec_dens for all values of self.fine_grid (rotated in complex plane)
        """
        # create numpy vectorized version of callable
    
        spec_dens_array_cmplx = self.spec_dens(self.fine_grid * np.exp(1.0j * self.phi))

        return spec_dens_array_cmplx

    def get_shared_attributes(self) -> dict:
        """
        Returns all attributes from the KernelMatrix base class.
        This is useful when initializing one of the inherited classes with
        an instance of another inherited class.

        Returns:
            dict: Dictionary containing all class attributes of KernelMatrix.
        """
        base_class_attributes = [
            "m",
            "n",
            "beta",
            "N_max",
            "delta_t",
            "h",
            "phi",
            "times",
            "fine_grid",
            "k_values",
            "kernel",
            "spec_dens",
            "spec_dens_array_fine",
            "freq_parametrization",
        ]

        base_class_attrs = {
            key: getattr(self, key, None) for key in base_class_attributes
        }

        return base_class_attrs

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_keys = ["m", "n", "beta", "N_max", "delta_t", "h", "phi", "freq_parametrization"]

        param_dict = {key: getattr(self, key) for key in param_keys}

        return param_dict

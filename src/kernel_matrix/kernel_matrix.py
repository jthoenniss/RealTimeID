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
        - only_positive_particle (bool): If True, only the positive frequencies for the particle component are included.
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
        only_positive_particle: bool,
        **kwargs
    ):
        # check if all parameters are valid
        KernelParams.validate_m_n(m, n)
        KernelParams.validate_beta(beta)
        KernelParams.validate_N_max_and_delta_t(N_max, delta_t)
        KernelParams.validate_h(h)
        KernelParams.validate_phi(phi)
        KernelParams.validate_freq_parametrization(freq_parametrization)

        for kwarg in kwargs:# ignore if an upper cutoff is specified as this is only relevant when computing continuous frequency integral as in DiscrKernel
            if kwarg == "upper_cutoff":
                pass
            else:
                raise ValueError(f"Invalid keyword argument: {kwarg}")
    

        # Store parameters
        self.m, self.n = m, n
        self.beta = beta
        self.N_max = N_max
        self.delta_t = delta_t
        self.h = h
        self.phi = phi
        self.spec_dens = spec_dens
        self.freq_parametrization = freq_parametrization
        self.only_positive_particle = only_positive_particle

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
        self.fine_grid, self.k_values, jacobian = cf.initialize_fine_grid(self.m, self.n, self.h, self.freq_parametrization)
        # initialize matrix kernel
        self.kernel = self._initialize_kernel(jacobian=jacobian)
        # initialize the spectral density as 1 for all values of the fine grid (spec_dens is included in the kernel matrix)
        self.spec_dens_array_fine = np.ones_like(self.kernel[0,:])



    def _initialize_kernel(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Creates the kernel matrix using the Fermi distribution function and spectral density.
        Parameters:
        - jacobian (np.ndarray): The Jacobian of the transformation from the measure: dw/dk.

        Returns:
        - np.ndarray: Kernel matrix.
        """
        times_arr = self.times[:, np.newaxis]  # enable broadcasting
        fine_grid_complex_positive = self.fine_grid * np.exp(1.0j * self.phi)
        jacobian_cmplx_positive = jacobian * np.exp(1.j * self.phi)#adjust for complex contour
        
        if self.only_positive_particle: #if only positive frequencies of the particle component are included
            # Kernel defined by Fermi distribution and spectral density
            #particle component
            K_particle = cf.dynamic_distr_particle(times_arr, fine_grid_complex_positive, self.beta) * self.spec_dens(fine_grid_complex_positive)
    
            return K_particle * jacobian_cmplx_positive

        # add negative frequencies
        fine_grid_complex = np.concatenate((-fine_grid_complex_positive[::-1].conj(), fine_grid_complex_positive))
        jacobian_cmplx = np.concatenate((jacobian_cmplx_positive[::-1], jacobian_cmplx_positive)) # add negative frequencies

        # Kernel defined by Fermi distribution and spectral density
        #particle component
        K_particle = cf.dynamic_distr_particle(times_arr, fine_grid_complex, self.beta) * self.spec_dens(fine_grid_complex)
        #hole component (negative sign in beta for hole distribution)
        K_hole = cf.dynamic_distr_particle(times_arr, fine_grid_complex, - self.beta) * self.spec_dens(fine_grid_complex)

        # Combine particle and hole contributions by stacking them on top of each other
        K = np.vstack((K_particle, K_hole))

        K *= jacobian_cmplx # multiply by Jacobian from measure. This is dw/dk. Multiply by exp(i*phi) to rotate in the complex plane

        return K


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
            "only_positive_particle",
        ]

        base_class_attrs = {
            key: getattr(self, key, None) for key in base_class_attributes
        }

        return base_class_attrs

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_keys = ["m", "n", "beta", "N_max", "delta_t", "h", "phi", "freq_parametrization", "only_positive_particle"]

        param_dict = {key: getattr(self, key) for key in param_keys}

        return param_dict
    
    def discrete_integral(
        self, kernel: np.ndarray = None, spec_dens_array_fine: np.ndarray = None
    ) -> np.ndarray:
        """
        Computes the discrete approximation to the frequency integral at the times defined on the time grid

        Parameters:
        - kernel (np.ndarray, optional): Kernel matrix, where different rows correspond to different time steps, and different columns correspond to different frequencies
        - spec_dens_array_fine (np.ndarray, optional): Array of spectral density values at the frequency points in the complex plane

        Returns:
        - np.ndarray: Discrete approximation result to frequency integral at times on time grid
        """

        # Use provided or default kernel and spec_dens_array
        kernel_eff = self.kernel if kernel is None else kernel
        spec_dens_array_eff_cmplx = (
            self.spec_dens_array_fine
            if spec_dens_array_fine is None
            else spec_dens_array_fine
        )

        if not isinstance(kernel_eff, np.ndarray):
            raise TypeError(
                f"'kernel' must be of type np.ndarray. Found {type(kernel_eff).__name__}"
            )

        if not isinstance(spec_dens_array_eff_cmplx, np.ndarray):
            raise TypeError(
                f"'spec_dens_array' must be of type np.ndarray. Found {type(spec_dens_array_eff_cmplx).__name__}"
            )
        if kernel_eff.shape[1] != len(spec_dens_array_eff_cmplx):
            raise RuntimeError(
                f"Frequency dimension of 'kernel' must match length of 'spec_dens_array'. Respective values found: {kernel_eff.shape[1]}, {len(spec_dens_array_fine)}"
            )

        # Sum over the frequency axis
        right_segment = kernel_eff @ spec_dens_array_eff_cmplx

        return right_segment

import numpy as np
from src.utils import common_funcs as cf
from src.utils.parameter_validator import ParameterValidator

class KernelMatrix:
    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        N_max: int,
        delta_t: float,
        h: float,
        phi: float
    ):
        """
        Initialize parameters for creating a kernel matrix.

        Parameters:
        - m (int): Number of discretization intervals for omega > 1/e.
        - n (int): Number of discretization intervals for omega < 1/e.
        - beta (float): Inverse temperature.
        - N_max (int): number of points on time grid
        - delta_t (float): time step
        - h (float): Discretization parameter.
        - phi (float): Rotation angle in the complex plane.
        """

        # check if all parameters are valid
        ParameterValidator.validate_m_n(m, n)
        ParameterValidator.validate_beta(beta)
        ParameterValidator.validate_N_max_and_delta_t(N_max, delta_t)
        ParameterValidator.validate_h(h)
        ParameterValidator.validate_phi(phi)
       
        # Store attributes
        self.m, self.n = m, n
        self.beta = beta
        self.N_max = N_max
        self.delta_t = delta_t
        self.h = h
        self.phi = phi

        #set time grid
        self.times = cf.set_time_grid(N_max = N_max, delta_t= delta_t)
        #initialize frequency grid
        self.fine_grid, self.k_values = self._initialize_fine_grid()
        #initialize matrix kernel
        self.kernel = self._initialize_kernel()

    def _initialize_fine_grid(self) -> np.ndarray:
        """
        Generates a fine grid for given discretization parameters.

        Returns:
        - np.ndarray: The generated fine grid as a NumPy array.
        - np.ndarray: The values for k that define the points on the grid.
        """
        k_values = np.arange(-self.n, self.m + 1)
        fine_grid = np.exp(self.h * k_values - np.exp(-self.h * k_values))
        return fine_grid, k_values

    def _initialize_kernel(self) -> np.ndarray:
        """
        Creates the kernel matrix using the Fermi distribution function and spectral density.

        Returns:
        - np.ndarray: Kernel matrix.
        """
        times_arr = self.times[:, np.newaxis] # enable broadcasting
        fine_grid_complex = self.fine_grid * np.exp(1.0j * self.phi)

        # Kernel defined by Fermi distribution, multiplied by spectral density
        K = cf.distr(times_arr, fine_grid_complex, self.beta) * cf.spec_dens_array(fine_grid_complex)
        K *= self.h * (1 + np.exp(-self.h * self.k_values)) * fine_grid_complex #factors from measure

        return K
    
    def _update_kernel(self):
        #update time grid
        self.times = cf.set_time_grid(N_max = self.N_max, delta_t= self.delta_t)
        #update frequency grid
        self.fine_grid, self.k_values = self._initialize_fine_grid()
        #update matrix kernel
        self.kernel = self._initialize_kernel()

    def get_shared_attributes(self):
        """
        Returns all attributes from the class.
        This is useful when initializing one of the 
        inherited classes with an instance of another inherited class.

        Returns:
        - dict: Dictionary containing all class attributes
        """
        return vars(self)



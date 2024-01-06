import numpy as np
from src.utils import common_funcs as cf
from src.kernel_params.parameter_validator import ParameterValidator

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
    ):
       
        # check if all parameters are valid
        ParameterValidator.validate_m_n(m, n)
        ParameterValidator.validate_beta(beta)
        ParameterValidator.validate_N_max_and_delta_t(N_max, delta_t)
        ParameterValidator.validate_h(h)
        ParameterValidator.validate_phi(phi)
       
        # Store parameters
        self.m, self.n = m, n
        self.beta = beta
        self.N_max = N_max
        self.delta_t = delta_t
        self.h = h
        self.phi = phi
        self.spec_dens = spec_dens

        #compute the kernel and time/frequency/spec_dens grids
        self._initialize_kernel_and_grids()
    
    
    def _initialize_kernel_and_grids(self) -> None:
        """
        (Re)compute the time grid, fine frequency grid, the kernel matrix and the vectorized spectrald density. 
        Needed for initialization and after change of parameters.
        """
        #set time grid
        self.times = cf.set_time_grid(N_max = self.N_max, delta_t= self.delta_t)
        #initialize frequency grid
        self.fine_grid, self.k_values = self._initialize_fine_grid()
        #initialize matrix kernel
        self.kernel = self._initialize_kernel()
        #initialize the spectral density evaluated at the grid points (rotated in complex plane)
        self.spec_dens_array_cmplx = self._compute_vectorized_spec_dens()

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

        # Kernel defined by Fermi distribution
        K = cf.distr(times_arr, fine_grid_complex, self.beta)
        K *= self.h * (1 + np.exp(-self.h * self.k_values)) * fine_grid_complex #factors from measure

        return K

    def _compute_vectorized_spec_dens(self) -> np.ndarray:
        """
        Helper function for vecotized computation 

        Returns:
        np.ndarray: array with values of spec_dens for all values of self.fine_grid (rotated in complex plane)
        """
        #create numpy vectorized version of callable
        spec_dens_vectorized = np.vectorize(self.spec_dens)

        return spec_dens_vectorized(self.fine_grid * np.exp(1.j * self.phi))

    def get_shared_attributes(self) -> dict:
        """
        Returns all attributes from the KernelMatrix base class.
        This is useful when initializing one of the inherited classes with
        an instance of another inherited class.

        Returns:
            dict: Dictionary containing all class attributes of KernelMatrix.
        """
        base_class_attributes = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel", "spec_dens", "spec_dens_array_cmplx"]
        
        base_class_attrs = {key: getattr(self, key, None) for key in base_class_attributes}

        return base_class_attrs
    
    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_keys = ["m", "n", "beta", "N_max", "delta_t", "h", "phi"]

        param_dict = {key: getattr(self, key) for key in param_keys}

        return param_dict
        
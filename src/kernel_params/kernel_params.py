import numpy as np
import math  # for floor and ceiling
from typing import Any #for type hinting

class KernelParams:
    MAX_EPS = 1.0
    UPPER_CUTOFF_ARGUMENT_DISCRETE_DEFAULT = 15.
    LOWER_CUTOFF_ARGUMENT_DISCRETE_DEFAULT = 3.5
    """
    A class designed to encapsulate and manage parameters essential for specifying a kernel matrix
    and its associated continuous integration results.

    This class serves as a container for key parameters used in kernel matrix calculations, such as
    discretization intervals, inverse temperature, time grid settings, discretization parameter, and
    rotation angle in the complex plane.

    It provides a number of static functions to validate parameter values

    Parameters:
        - m (int): Number of discretization intervals for frequencies greater than 1/e.
        - n (int): Number of discretization intervals for frequencies less than 1/e.
        - N_max (int): Maximum number of points on the time grid.
        - delta_t (float): Time step for the time grid.
        - beta (float): Inverse temperature, a crucial thermodynamic parameter.
        - upper_cutoff (float): Frequency cutoff for continuous integrations.
        - upper_cutoff_argument_discrete (float): Frequency cutoff for discrete integrations.
        - lower_cutoff_argument_discrete (float): Frequency cutoff for discrete integrations.
        - h (float): Discretization parameter, influencing frequency grid resolution.
        - phi (float): Rotation angle in the complex frequency plane.
        - spec_dens (callable): Single-parameter function returning the spectral density.
        

    Attributes:
        _params (dict): A dictionary that holds the parameter values.

    Methods:
        params() -> dict: Returns the dictionary of stored parameters.
        get(key: str) -> Any: Retrieves the value of a specific parameter using its key.
        update_parameters(updates: dict) -> None: Updates multiple parameters simultaneously based on a dictionary of updates.

        + a list of static methods for parameter validation
    """
    
    def __init__(self, **kwargs):
        
        #Initialize dictionary of parameters to default values
        self._params = {
            "m": 10,
            "n": 10,
            "N_max": 10,
            "delta_t": 0.1,
            "beta": np.inf,
            "upper_cutoff": np.inf, 
            "h": 0.1,
            "phi": np.pi / 4,
            "spec_dens": lambda x: 1.
        }    
        
        # For those parameters that are specified as keyword arguments, update the values
        for key, value in kwargs.items():
            self._set_param(key, value)


        # Set the cutoff arguments for the discrete integrals to default value:    
        self.set_discrete_cutoffs()

    @property
    def params(self) -> dict:
        """
        Return the dictionary of parameters that are stored as attributes
        """
        return self._params
    

    
    def get_param(self, key: str) -> Any:
        """
        Get the value of a parameter by providing its key
        """

        if key not in self._params.keys():
            raise KeyError(f"Key {key} not in parameter list.")

        return self._params.get(key)
    
    def _set_param(self, key: str, value: Any) -> None:
        """
        Set the value of a parameter by providing its key and value
        - key (str): The key of the parameter to be set
        - value (Any): The value to which the parameter should be set
        
        Returns:
        - None

        Raises: 
        - KeyError: If the key is not in the parameter list
        """
        if key in self._params:
                self._params[key] = value #set parameter value
        else:
            raise KeyError(f"Key {key} not in parameter list. Valid parameter keys are: {self._params.keys()}.")

    def update_parameters(self, updates: dict) -> None:
        """
        Update grid parameters with new values.

        Parameters:
        - updates (dict): Dictionary containing parameter names and their new values.

        Returns:
        - None

        Note:
        - A warning is issued if 'h', 'm', and 'n' parameters are attempted to be updated simultaneously, 
          as this may lead to unintended consequences in the grid configuration.

        Raises:
        - TypeError: If the input is not a dictionary.
        """

        if not isinstance(updates, dict):
            raise TypeError(f"Input must be of type dict, not {type(updates)}.")
        
        # Check for simultaneous updates of 'h', 'm', and 'n' and issue warning if necessary
        if "h" in updates:
            intersection = set(updates).intersection({"m", "n"})#get intersection of keys in updates and keys "m", "n"
            if intersection: #if intersection is not empty
                print(f"Warning: Simultaneous updates to parameters 'h' and {intersection} detected. Please double-check for consistency.")

        for key, value in updates.items():
            
            self._set_param(key, value)#set parameter value

            if key == "h":  # Special handling when 'h' is updated
                # Update 'm' and 'n' based on the new value of 'h'
                lower_cutoff_argument_discrete, upper_cutoff_argument_discrete = self.get_discrete_cutoffs()
                self._set_param("m", math.ceil(upper_cutoff_argument_discrete / value)) 
                self._set_param("n", math.ceil(lower_cutoff_argument_discrete / value)) 

    def set_discrete_cutoffs(self, lower_cutoff_argument_discrete: float = None, upper_cutoff_argument_discrete: float = None) -> None:
        """
        Set the values of the discrete cutoffs for the frequency grid.
        - lower_cutoff_argument_discrete (float, optional): The lower cutoff for the frequency grid.
        - upper_cutoff_argument_discrete (float, optional): The upper cutoff for the frequency grid.

        Returns:
        - None
        """
        #Choose w, such that exp(-w - exp(w)) = 1.e-16 or lower
        if lower_cutoff_argument_discrete is not None:
            self.validate_lower_cutoff_argument_discrete(lower_cutoff_argument_discrete)
            self._lower_cutoff_argument_discrete = lower_cutoff_argument_discrete
        else:
            self._lower_cutoff_argument_discrete = KernelParams.LOWER_CUTOFF_ARGUMENT_DISCRETE_DEFAULT

        #Choose w, such that exp(w - exp(-w)) = 1.e6 or higher
        if upper_cutoff_argument_discrete is not None:   
            self.validate_upper_cutoff_argument_discrete(upper_cutoff_argument_discrete)
            self._upper_cutoff_argument_discrete = upper_cutoff_argument_discrete
        else:
            self._upper_cutoff_argument_discrete = KernelParams.UPPER_CUTOFF_ARGUMENT_DISCRETE_DEFAULT  

    def get_discrete_cutoffs(self) -> tuple:
        """
        Get the values of the discrete cutoffs for the frequency grid.
        """
        return self._lower_cutoff_argument_discrete, self._upper_cutoff_argument_discrete

    @staticmethod
    def validate_m_n(m: int, n: int):
        if not isinstance(m, int) or m < 0:
            raise ValueError(f"'m' must be a non-negative integer, got {m}")
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"'n' must be a non-negative integer, got {n}")

    @staticmethod
    def validate_beta(beta: float):
        if not isinstance(beta, (float, int)) or beta < 0:
            raise ValueError(f"'beta' must be a positive number, got {beta}")

    @staticmethod
    def validate_times(times: np.ndarray):
        if not isinstance(times, np.ndarray) or times.ndim != 1 or times.size == 0:
            raise ValueError("Attribute 'times' must be a non-empty 1D NumPy array.")
   

    @staticmethod
    def validate_N_max_and_delta_t(N_max: int, delta_t: float):
        if not isinstance(N_max, (int, np.int64)):
            raise ValueError(f"Attribute 'N_max' must be an int, got {type(N_max).__name__}.")

        if not isinstance(delta_t, (float, int, np.float64)):
            raise ValueError(f"Attribute 'delta_t' must be a float or int, got {type(delta_t).__name__}.")

        if N_max <= 0:
            raise ValueError(f"Attribute 'N_max' must be positive, got {N_max}.")

        if delta_t <= 0:
            raise ValueError(f"Attribute 'delta_t' must be positive, got {delta_t}.")

            
    @staticmethod
    def validate_eps(eps: float):
        if not 0 < eps < KernelParams.MAX_EPS:
            raise ValueError(f"'eps' must be between 0 and {KernelParams.MAX_EPS}, got {eps}")

    @staticmethod
    def validate_h(h: float):
        if not isinstance(h, (float, int)) or h <= 0:
            raise ValueError(f"'h' must be a positive number, got {h}")

    @staticmethod  
    def validate_phi(phi: float):
        if not isinstance(phi, (float, int)):
            raise ValueError(f"'phi' must be a number, got {phi}")

    @staticmethod  
    def validate_upper_cutoff(upper_cutoff: float):
        if not isinstance(upper_cutoff, (float, int)) or upper_cutoff <= 0:
            raise ValueError(f"'upper_cutoff' must be a positive number, got {upper_cutoff}")
        
    @staticmethod  
    def validate_upper_cutoff_argument_discrete(upper_cutoff_argument_discrete: float):
        if not isinstance(upper_cutoff_argument_discrete, (float, int)) or upper_cutoff_argument_discrete <= 0:
            raise ValueError(f"'upper_cutoff' must be a positive number, got {upper_cutoff_argument_discrete}")
        
    @staticmethod  
    def validate_lower_cutoff_argument_discrete(lower_cutoff_argument_discrete: float):
        if not isinstance(lower_cutoff_argument_discrete, (float, int)) or lower_cutoff_argument_discrete <= 0:
            raise ValueError(f"'lower_cutoff' must be a positive number, got {lower_cutoff_argument_discrete}")
        

    @staticmethod
    def validate_required_params(params, required_params):
        """
        Validate if all parameters specifided in 'required_parameters' are present.

        Parameters:
            params (dict): Parameters to validate.
            required_params (list): List of required parameter names.
        
        Raises:
            ValueError: If any required parameter is missing.
        """
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(f"Missing required parameter(s): {', '.join(missing_params)}")

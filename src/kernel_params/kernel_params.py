import numpy as np
import math  # for floor and ceiling
from typing import Any #for type hinting

class KernelParams:

    """
    A class designed to encapsulate and manage parameters essential for specifying a kernel matrix
    and its associated continuous integration results.

    This class serves as a container for key parameters used in kernel matrix calculations, such as
    discretization intervals, inverse temperature, time grid settings, discretization parameter, and
    rotation angle in the complex plane.

    Parameters:
        - m (int): Number of discretization intervals for frequencies greater than 1/e.
        - n (int): Number of discretization intervals for frequencies less than 1/e.
        - N_max (int): Maximum number of points on the time grid.
        - delta_t (float): Time step for the time grid.
        - beta (float): Inverse temperature, a crucial thermodynamic parameter.
        - upper_cutoff (float): Frequency cutoff for continuous integrations.
        - upper_cutoff_discrete (float): Frequency cutoff for discrete integrations.
        - lower_cutoff_discrete (float): Frequency cutoff for discrete integrations.
        - h (float): Discretization parameter, influencing frequency grid resolution.
        - phi (float): Rotation angle in the complex frequency plane.
        - spec_dens (callable): Single-parameter function returning the spectral density.
        

    Attributes:
        _params (dict): A dictionary that holds the parameter values.

    Methods:
        params() -> dict: Returns the dictionary of stored parameters.
        get(key: str) -> Any: Retrieves the value of a specific parameter using its key.
        update_parameters(updates: dict) -> None: Updates multiple parameters simultaneously based on a dictionary of updates.
    """
    
    def __init__(self, **kwargs):

        #Initialize dictionary of parameters to default values
        self._params = {
            "m": None,
            "n": None,
            "N_max": None,
            "delta_t": 0.1,
            "beta": np.inf,
            "upper_cutoff": np.inf, 
            "upper_cutoff_discrete": 15.0, # with this choice, the frequency grid reaches up to 1.e6 or higher.
            "lower_cutoff_discrete": 3.6, # with this choice, the frequency grid reaches down to 1.e-16 or lower.
            "h": None,
            "phi": np.pi / 4,
            "spec_dens": lambda x: 1.
        }    

        # For those parameters that are specified as keyword arguments, update the values
        for key, value in kwargs.items():
            self._set(key, value)

    @property
    def params(self) -> dict:
        """
        Return the dictionary of parameters that are stored as attributes
        """
        return self._params
    
    def get(self, key: str) -> Any:
        """
        Get the value of a parameter by providing its key
        """

        if key not in self._params.keys():
            raise KeyError(f"Key {key} not in parameter list.")

        return self._params.get(key)
    
    def _set(self, key: str, value: Any) -> None:
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
            
            self._set(key, value)#set parameter value

            if key == "h":  # Special handling when 'h' is updated
                # Update 'm' and 'n' based on the new value of 'h'
                self._params["m"] = math.ceil(self.get("upper_cutoff_discrete") / value) #set value of m such that is reaches up to upper_cutoff_discrete for given h
                self._params["n"] = math.ceil(self.get("lower_cutoff_discrete") / value) #set value of n such that is reaches down to lower_cutoff_discrete for given h


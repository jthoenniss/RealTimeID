import numpy as np
import math  # for floor and ceiling

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
    
    def __init__(self):
        self._params = {
            "m": None,
            "n": None,
            "N_max": None,
            "delta_t": 0.1,
            "beta": np.inf,
            "upper_cutoff": np.inf,
            "h": None,
            "phi": np.pi / 4,
            "spec_dens": lambda x: 1.
        }    

    @property
    def params(self) -> dict:
        """
        Return the dictionary of parameters that are stored as attributes
        """
        return self._params
    
    def get(self, key: str):
        """
        Get the value of a parameter by providing its key
        """
        return self._params.get(key)

    def update_parameters(self, updates):
        """
        Update grid parameters with new values.

        Parameters:
        - updates (dict): Dictionary containing parameter names and their new values.

        Returns:
        - None
        """

        for name, value in updates.items():

            self._params[name] = value
            if name == "h":
                # when h is varied, also update m and n
                self._params["m"] = math.ceil(15.0 / value) # with this choice, the frequency grid reaches up to 1.e8 or higher.
                self._params["n"] = math.ceil(3.6 / value) # with this choice, the frequency grid reaches down to 1.e-16 or lower.

        # If m and n are varied simultaneously, throw warning.
        updated_params = set(updates.keys())
        intersection = updated_params.intersection({"h", "m", "n"})
        if len(intersection) > 1:
            print(
                f"Warning: Attempting to update the parameters {intersection} simultaneously. Double check that this is intended."
            )

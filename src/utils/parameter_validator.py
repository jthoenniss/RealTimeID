import numpy as np

class ParameterValidator:
    MAX_EPS = 1.0
    """
    Validate the parameters of the RtKernel class.

    Raises:
        ValueError: If any parameter is out of its expected range or type.
    """
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
        if not 0 < eps < ParameterValidator.MAX_EPS:
            raise ValueError(f"'eps' must be between 0 and {ParameterValidator.MAX_EPS}, got {eps}")

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
    def validate_required_params(params, required_params):
        """
        Validate if all required parameters are present.

        Parameters:
            params (dict): Parameters to validate.
            required_params (list): List of required parameter names.
        
        Raises:
            ValueError: If any required parameter is missing.
        """
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(f"Missing required parameter(s): {', '.join(missing_params)}")

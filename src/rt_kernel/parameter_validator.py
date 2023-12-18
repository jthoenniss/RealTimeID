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
        if not isinstance(beta, (float, int)) or beta <= 0:
            raise ValueError(f"'beta' must be a positive number, got {beta}")

    @staticmethod
    def validate_times(times: np.ndarray):
        if not isinstance(times, np.ndarray) or times.ndim != 1 or times.size == 0:
            raise ValueError("Attribute 'times' must be a non-empty 1D NumPy array.")

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

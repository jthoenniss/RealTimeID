import numpy as np

class DataInterp:
    """
    Class that interpolates data, given on a discrete, arbitrarily spaced grid, 
    to extract a continuous function. Supports both linear and logarithmic 
    interpolation scales.
    """

    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray) -> None:

        argsort = np.argsort(x_vals)
        self.x_vals = x_vals[argsort]
        self.y_vals = y_vals[argsort]

    def interp(self, x_eval: float, x_scale: str = 'lin') -> float:
        """
        Interpolates the data at the specified point.

        Parameters:
        - x_eval (float or np.ndarray): Point(s) at which the data should be evaluated.
        - x_scale (str): Set to 'lin' for linear or 'log' for logarithmic interpolation.

        Returns:
        - Interpolated value(s) (float or np.ndarray) if x_eval is within the range, else 0.0.
        """

        if x_scale not in ['lin', 'log']:
            raise ValueError("x_scale must be either 'lin' or 'log'.")

        if x_scale == 'log':
            if np.any(x_eval <= 0):
                raise ValueError("x_eval must be positive for logarithmic interpolation.")

            x_eval_log = np.log(x_eval)
            x_vals_log = np.log(self.x_vals)

            return np.interp(x_eval_log, x_vals_log, self.y_vals, left=0.0, right=0.0)
     
        else:  # Linear interpolation
            return np.interp(x_eval, self.x_vals, self.y_vals, left=0.0, right=0.0)

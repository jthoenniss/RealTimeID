import numpy as np

def create_numpy_arrays(D):
    """
    Create NumPy arrays from a list of objects.

    Parameters:
    - D (list): List of objects with attributes eps, m, n, h.

    Returns:
    - Tuple of NumPy arrays (errors, m_vals, n_vals, h_vals).
    """
    data = [(d.eps, int(d.m), int(d.n), d.h) for d in D]
    dtypes = [float, int, int, float]

    return tuple(np.array(arr, dtype=dtype) for arr, dtype in zip(zip(*data), dtypes))


def check_error_condition(eps_current, eps_previous):
    """
    Check the error condition to determine if the iteration should be stopped.

    Parameters:
    - eps_current (float): Current error value.
    - eps_previous (float): Previous error values.

    Returns:
    - bool: True if the condition is met, indicating the iteration should be stopped; otherwise, False.
    """
    if eps_current > eps_previous or eps_current < 1.0e-15:
        print(
            f"Either the error does not shrink with decreasing h or the machine precision error was reached. "
            f"Error = {eps_current}. Stopping the iteration."
        )
        return True
    return False



def update_grid_parameters(params, h):
    """
    Update grid parameters in the given dictionary.

    Parameters:
    - params (dict): Dictionary containing grid parameters.
    - h (float): New value for the discretization parameter 'h'.
    """
    params["h"] = h
    params["m"] = int(10.0 / h)
    params["n"] = int(5.0 / h)

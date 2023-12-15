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

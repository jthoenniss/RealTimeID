import numpy as np
import math  # for floor and ceiling
from scipy import integrate


def create_numpy_arrays_from_kernel(D):
    """
    Create NumPy arrays from a list of objects.

    Parameters:
    -  D (np.ndarray): Structured array of objects with attributes eps, m, n, h, ID_rank.

    Returns:
    - Tuple[dict, np.ndarray]: Tuple containing a dictionary with data arrays, corresponding to different quantities at all parameter combinations, and np.ndarray which holds the kernel dimensions.
    """
    # Retain original shape of D
    kernel_dims = np.array(D.shape)

    # Flatten D for convenience
    D = D.flatten()

    data = [(d.eps, d.m, d.n, d.h, d.ID_rank, d.N_max, d.beta, d.delta_t) for d in D]
    dtypes = [float, int, int, float, int, int, float, float]

    errors, m_vals, n_vals, h_vals, ID_ranks, N_maxs, betas, delta_t_vals = tuple(
        np.array(arr, dtype=dtype) for arr, dtype in zip(zip(*data), dtypes)
    )

    

    return (
            {
                "errors": errors.reshape(kernel_dims),
                "m_vals": m_vals.reshape(kernel_dims),
                "n_vals": n_vals.reshape(kernel_dims),
                "h_vals": h_vals.reshape(kernel_dims),
                "betas": betas.reshape(kernel_dims),
                "N_maxs": N_maxs.reshape(kernel_dims),
                "ID_ranks": ID_ranks.reshape(kernel_dims),
                "delta_t_vals": delta_t_vals.reshape(kernel_dims),
            },
            kernel_dims,
        )



def check_error_condition(eps_current, eps_previous):
    """
    Check the error condition to determine if the iteration should be stopped.

    Parameters:
    - eps_current (float): Current error value.
    - eps_previous (float): Previous error values.

    Returns:
    - bool: True if the condition is met, indicating the iteration should be stopped; otherwise, False.
    """
    if eps_current > eps_previous or eps_current < 1.0e-14:
        print(
            f"Either the error does not shrink with decreasing h or the machine precision error was reached. "
            f"Error = {eps_current}. Stopping the iteration."
        )
        return True
    return False


def update_parameters(params, updates):
    
    #Update grid parameters with new values.

    #Parameters:
    #- params (dict): Dictionary containing grid parameters.
    #- updates (dict): Dictionary containing parameter names and their new values.

    #Returns:
    #- None
    

    for name, value in updates.items():
        params[name] = value
        if name == "h":
            # when h is varied, also update m and n
            params["m"] = math.ceil(15.0 / value) # with this choice, the frequency grid reaches up to 1.e6 or higher.
            params["n"] = math.ceil(3.6 / value) # with this choice, the frequency grid reaches down to 1.e-16 or lower.

    # If m and n are varied simultaneously, throw warning.
    updated_params = set(updates.keys())
    intersection = updated_params.intersection({"h", "m", "n"})
    if len(intersection) > 1:
        print(
            f"Warning: Attempting to update the parameters {intersection} simultaneously. Double check that this is intended."
        )


def distr(t, x, beta: float):
    """
    Compute time-dependent Kernel, e^{i*t*omega} * (1-n_F(omega)), where n_F is Fermi-Dirac distribution
    Note: omega is parametrized as x * e^{i*phi}, where x is real

    Parameters:
    - t (float): time argument
    - x (float/complex): frequency argument
    - beta (float): inverse temperature

    Returns:
    int: Kernel evaluated at specified paramters
    """
    return np.exp(1.0j * x * t) / (1 + np.exp(-beta * x))


#def spec_dens_scalar(omega_scalar):
#    """
#    Compute spectral density for a single frequency point (float or complex).

#    Parameters:
#    - omega_float (float or complex): Single frequency point

#    Returns:
#    - float : Spectral density evaluated at this frequency point
#    """
#    return 1.0



def compute_singular_values(matrix, relative_error):
    """
    Compute the Singular Value Decomposition (SVD) rank of a matrix.

    Parameters:
    - matrix (numpy.ndarray): Input matrix for which the SVD rank is computed.
    - relative_error (float): Desired relative error threshold.

    Returns:
    int: The computed SVD rank based on the specified relative error.
    """
    # Perform SVD
    _, singular_values, _ = np.linalg.svd(matrix)

    # Compute the total sum of squared singular values
    total_sum = np.sum(singular_values)

    svd_rank = np.sum([singular_values / total_sum > relative_error])

    return svd_rank, singular_values


def generate_chebyshev_grid_in_interval(a, b, m):
    """
    Generate a Chebyshev grid of order m within the interval [a, b].

    Parameters:
    - a (float): Start of the interval.
    - b (float): End of the interval.
    - m (int): Order of the Chebyshev grid.

    Returns:
    numpy.ndarray: Chebyshev nodes within the interval.
    """
    k_values = np.arange(1, m + 1)
    chebyshev_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
        (2 * k_values - 1) * np.pi / (2 * m)
    )
    return np.sort(chebyshev_nodes)


def generate_composite_chebyshev_grid_dyadic(M_intervals, m_chebyshev, upper_cutoff):
    """
    Generate a composite Chebyshev grid with Chebyshev nodes in each interval.
    Intervals are dyadically refined towards origin.

    Parameters:
    - M_intervals (int): Number of intervals in the composite grid.
    - m_chebyshev (int): Order of the Chebyshev grid in each interval.

    Returns:
    numpy.ndarray: Composite Chebyshev grid with nodes.
    """
    cheb_points = []

    for i in range(1, M_intervals + 1):
        a_i = 0.0 if i == 1 else 1 / 2 ** (M_intervals - i + 1)
        b_i = 1.0 if i == M_intervals else 1 / 2 ** (M_intervals - i)

        # Generate Chebyshev nodes in the interval [a_i, b_i]
        cheb_nodes_in_interval = generate_chebyshev_grid_in_interval(
            a_i, b_i, m_chebyshev
        )
        cheb_points.extend(cheb_nodes_in_interval)

    return upper_cutoff * np.array(cheb_points)


def set_time_grid(N_max, delta_t):
    """
    Initializes the dicrete-time grid for fixed final time and time step
    Parameters:
    - N_max (int): nbr. of time steps up to final time
    - delta_t (float): time step

    Returns:
    np.array(): array containing the time points
    """
    return np.arange(1, N_max + 1) * delta_t


def cont_integral(t, beta, upper_cutoff, spec_dens: callable, phi=np.pi / 4):
    """
    Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff], at fixed time t
    Parameters:
    - t (float or array-like): Time argument(s)
    - beta (float): Inverse temperature
    - upper_cutoff (float): Energy upper_cutoff up to which kernel is integrated
    - spec_dens (callable): One-parameter function that returns the spectral density.
    - phi (float): Rotation angle in the complex plane

    Returns:
    - (np.complex_ or np.ndarray): Result(s) of integration in interval [0, upper_cutoff]
    """
    # Ensure t is an array
    t = np.atleast_1d(t)

    # Function to integrate for real and imaginary parts
    def integrand_real(x):
        return np.real(
            np.exp(1.0j * phi)
            * distr(t, x * np.exp(1.j * phi), beta)
            * spec_dens(x * np.exp(1.0j * phi))
        )

    def integrand_imag(x):
        return np.imag(
            np.exp(1.0j * phi)
            * distr(t, x * np.exp(1.j * phi), beta)
            * spec_dens(x * np.exp(1.0j * phi))
        )

    # Vectorized integration for real and imaginary parts
    right_segment_cont_real, _ = integrate.quad_vec(
        integrand_real,
        0,
        upper_cutoff,
        epsabs=1.49e-15,
        epsrel=1.49e-13
    )

    right_segment_cont_imag, _ = integrate.quad_vec(
        integrand_imag,
        0,
        upper_cutoff,
        epsabs=1.49e-15,
        epsrel=1.49e-13
    )

    return right_segment_cont_real + 1.0j * right_segment_cont_imag


def point_density(grid, lower_limit, upper_limit, interval_spacing="lin"):
    """
    Calculate point density within specified intervals.

    Parameters:
        grid (numpy.ndarray): Input data array.
        lower_limit: Lower limit of the interval. For interval_spacing = 'log', specify exponent, i.e. lower limit is then 10**lower_limit.
        upper_limit: Upper limit of the interval. For interval_spacing = 'log', specify exponent, i.e. upper limit is then 10**upper_limit.
        interval_spacing (str): Type of interval spacing ('lin' or 'log').

    Returns:
        numpy.ndarray: Array containing point density within each interval.
        numpy.ndarray: Array containing the midpoints of the intervals in which the density is evaluated
    """
    if interval_spacing == "lin":
        limits = np.array([math.floor(lower_limit), math.ceil(upper_limit)])
        point_density = np.array(
            [np.sum((grid >= a) & (grid < (a + 1)) for a in limits)]
        )
        point_density_grid = np.array([a + 0.5 for a in limits])

    elif interval_spacing == "log":
        assert isinstance(lower_limit, int) and isinstance(
            upper_limit, int
        ), "Lower and upper limit must be integers signifying the power of 10"
        del_a = 0.5
        limits = np.arange(lower_limit, upper_limit, del_a)
        point_density = np.array(
            [np.sum((grid >= 10.0**a) & (grid < 10.0 ** (a + del_a))) for a in limits]
        )
        point_density_grid = np.array(
            [(10.0**a + 10.0 ** (a + del_a)) / 2.0 for a in limits]
        )

    else:
        raise ValueError(
            "Invalid interval spacing parameter specified. Use 'lin' or 'log'."
        )

    return point_density, point_density_grid

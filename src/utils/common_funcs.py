import numpy as np
import math  # for floor and ceiling
from scipy import integrate


def create_numpy_arrays_from_kernel(D):
    """
    Create NumPy arrays from a list of objects.

    Parameters:
    -  D (np.ndarray): Structured array of objects with attributes eps, m, n, h, ID_rank.

    Returns:
    - Tuple[list, dict]: Tuple containing a list which holds the kernel dimensions, and a dictionary with data arrays, corresponding to different quantities at all parameter combinations
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

    

    return (kernel_dims, 
            {
                "eps": errors.reshape(kernel_dims),
                "m": m_vals.reshape(kernel_dims),
                "n": n_vals.reshape(kernel_dims),
                "h": h_vals.reshape(kernel_dims),
                "beta": betas.reshape(kernel_dims),
                "N_max": N_maxs.reshape(kernel_dims),
                "ID_rank": ID_ranks.reshape(kernel_dims),
                "delta_t": delta_t_vals.reshape(kernel_dims),
            },
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


def fermi_dirac(omega, beta: float, mu: float = 0):
    """
    Compute the Fermi-Dirac distribution function, 1 / (1 + exp(beta * (omega - mu))).
    """
    #handle errors due to division by zero (while errorstate)
    with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
        distr = 1 / (1 + np.exp(beta * (omega - mu)))
        # Set values where exp overflowed to 0
        distr = np.where(np.isfinite(distr), distr, 0)
    return distr

def dynamic_distr_particle(t, omega, beta: float, mu: float = 0): 
    """
    Compute time-dependent Kernel of the particle-propagator, e^{i*t*omega} * (1-n_F(omega)), where n_F is Fermi-Dirac distribution
    Note: 1) omega is parametrized as x * e^{i*phi}, where x is real
            2) to compute the hole propagator, set beta = -beta.

    Clip the arguments of the exponentials in order to avoid overflow/underflow errors.

    Parameters:
    - t (float): time argument
    - omega (float/complex): frequency argument
    - beta (float): inverse temperature
    - mu (float): chemical potential

    Returns:
    int: Kernel evaluated at specified paramters
    """
    # Safe limit for exponent in double precision
    max_exponent = 700

    # Compute the real and imaginary parts of omega * t
    omega_t_real = (omega * t).real
    omega_t_imag = (omega * t).imag

    # Clip the imaginary part of omega * t (real part gives just oscillation)
    clipped_omega_t_imag = np.clip(omega_t_imag, -max_exponent, max_exponent)
    clipped_omega_t = omega_t_real + 1.0j * clipped_omega_t_imag

    # Compute the exponentials with clipped arguments
    evolution = np.exp(1.0j * clipped_omega_t)
    distribution = fermi_dirac(omega, - beta, mu)#minus sign in beta results in HOLE distribution (which is relevant for PARTICLE propagator)
    
    return evolution * distribution


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
    - upper_cutoff (float): Scaling factor of the grid. The final composite grid reaches up to upper_cutoff.

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


def cont_integral(t, beta, upper_cutoff, spec_dens: callable, phi=np.pi / 4, only_positive: bool = False):
    """
    Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff], at fixed time t
    Parameters:
    - t (float or array-like): Time argument(s)
    - beta (float): Inverse temperature
    - upper_cutoff (float): Energy upper_cutoff up to which kernel is integrated
    - spec_dens (callable): One-parameter function that returns the spectral density.
    - phi (float): Rotation angle in the complex plane
    - only_positive (bool, optional): If True, only positive frequency branch is considered

    Returns:
    - (np.complex_ or np.ndarray): Result(s) of integration in interval [0, upper_cutoff]
    """
    # Ensure t is an array
    t = np.atleast_1d(t)

    #integrand, expressed as a sum of two parts where the first part refers to right segment of the contour and the second part to the left segment
    def integrand (omega): 
        freq = omega * np.exp(1.j * phi)
        positive_segment = dynamic_distr_particle(t, freq, beta) * spec_dens(freq) * np.exp(1.0j * phi)#exponential from jacobian

        if only_positive:
            return positive_segment  
        
        negative_segment = dynamic_distr_particle(t, -freq.conj(), beta) * spec_dens(- freq.conj()) * np.exp(-1.0j * phi)#exponential from jacobian
        return (positive_segment + negative_segment) 


    # Vectorized integration for real and imaginary parts
    integral, _ = integrate.quad_vec(
        integrand,
        0,
        upper_cutoff,
        epsabs=1.49e-15,
        epsrel=1.49e-13
    )


    return integral 



def initialize_fine_grid(m: int, n: int, h: float, freq_parametrization: str) -> tuple:
        """
        Generates a fine grid for given discretization parameters.
        Parameters:
        - m (int): Number of frequencies omega > 1.
        - n (int): Number of frequencies  0 < omega < 1.
        - h (float): Grid spacing.
        - freq_parametrization (str): Frequency parametrization ('simple_exp' or 'fancy_exp').

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - The generated fine grid as a NumPy array.
        - The k values that define the grid points.
        - The Jacobian of the transformation from the measure (dw/dk).
        """
        k_values = np.arange(-n, m + 1)

        #initialize fine grid and Jacobian with exact implementation depending on the grid parametrization
        if freq_parametrization == "simple_exp":
            fine_grid = np.exp(h * k_values)
            jacobian = h * fine_grid # Jacobian from measure. This is dw/dk.
            return fine_grid, k_values, jacobian

        elif freq_parametrization == "fancy_exp":
            fine_grid = np.exp(h * k_values - np.exp(-h * k_values))
            jacobian = h * (1 + np.exp(-h * k_values)) * fine_grid  # Jacobian from measure. This is dw/dk.
            return fine_grid, k_values, jacobian
        
        else:
            raise ValueError("Invalid grid parametrization argument. Must be 'simple_exp' or 'fancy_exp'. Got: " + freq_parametrization)

def time_integrate(time_series, delta_t):
        """
        Compute the time-integrated value based on a time-series.
        Parameters:
        - times_series(np.array(float)): time series to be integrated
        - delta_t(float): time step

        Returns:
        - float: time-integrated value
        """
        time_integrated_value = delta_t * np.sum(time_series)
        return time_integrated_value

def error_time_integrated(time_series_exact, time_series_approx, delta_t):
    """
    Compute the relative time-integrated error between two time series.

    Parameters:
    - time_series_exact (numpy.ndarray): Exact time series.
    - time_series_approx (numpy.ndarray): Approximated time series.
    - delta_t (float): Time step.

    Returns:
    - float: Relative time-integrated error.
    """

    # compute norm
    norm = time_integrate(abs(time_series_exact) + abs(time_series_approx), delta_t)

    #if time series is only numerical noise, do not compute relative error (which would be large although absolute values are essentially 0)
    if norm < 1.e-16 * len(time_series_exact):
        return 0

    #absolute time integrated error
    abs_error_time_integrated = time_integrate(abs(time_series_exact - time_series_approx), delta_t)
    
    # compute relative error by dividing by norm
    rel_error_time_integrated = abs_error_time_integrated / norm

    return rel_error_time_integrated


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

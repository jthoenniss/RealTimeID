import numpy as np
from scipy import integrate
import scipy.linalg.interpolative as sli


def distr(t, x, beta, phi=0):
    """
    Compute time-dependent Kernel, e^{i*t*omega} * (1-n_F(omega)), where n_F is Fermi-Dirac distribution
    Note: omega is parametrized as x * e^{i*phi}, where x is real

    Parameters:
    - t (float): time argument
    - x (float): frequency argument
    - beta (float): inverse temperature
    - phi (float): rotation angle in complex plane

    Returns:
    int: Kernel evaluated at specified paramters
    """
    return np.exp(1.0j * x * t * np.exp(1.0j * phi)) / (
        1 + np.exp(-beta * x * np.exp(1.0j * phi))
    )


def spec_dens(omega):
    """
    Compute spectral density as function of frequency

    Parameters:
    - frequency omega

    Returns:
    int: spectral density evaluated at this frequency
    """

    return 1


def svd_check_singular_values(matrix, relative_error):
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


def cont_integral(t, beta, upper_cutoff, phi=np.pi / 4):
    """
    Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff], at fixed time t
    Parameters:
    - t (float): time argument
    - beta (float): inverse temperature
    - upper_cutoff (float): energy upper_cutoff up to which kernel is integrated
    - phi (float): rotations angle in complex plane

    Returns:
    - (np.complex_): Result of integration in interval [0,upper_cutoff]
    """
    # compute real part by using integration routine
    right_segment_cont_real, _ = integrate.quad(
        lambda x: np.real(np.exp(1.0j * phi) * distr(t, x, beta, phi)),
        0,
        upper_cutoff,  # factor np.exp(1.j * phi) comes from integration measure
    )

    # compute imaginary part by using integration routine
    right_segment_cont_imag, _ = integrate.quad(
        lambda x: np.imag(np.exp(1.0j * phi) * distr(t, x, beta, phi)),
        0,
        upper_cutoff,  # factor np.exp(1.j * self.phi) comes from integration measure
    )

    return right_segment_cont_real + 1.0j * right_segment_cont_imag


class DiscrError:
    def __init__(self, m, n, N_max, delta_t, beta, upper_cutoff, phi=np.pi / 4, h=None):
        self.m = m
        self.n = n
        self.N_max = N_max
        self.delta_t = delta_t
        self.beta = beta
        self.upper_cutoff = upper_cutoff
        self.phi = phi
        self.times = set_time_grid(
            self.N_max, self.delta_t
        )  # set time grid with N_max time steps and time step delta_t

        if h is None:
            self.h = np.log(upper_cutoff) / self.m
        else:
            self.h = h

    def cont_integral(self, t):
        """
        Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff], at fixed time t
        Parameters:
        - t (float): time argument
        Returns:
        - (np.complex_): Result of integration in interval [0,upper_cutoff]
        """
        # compute real part by using integration routine
        right_segment_cont_real, _ = integrate.quad(
            lambda x: np.real(
                np.exp(1.0j * self.phi) * distr(t, x, self.beta, self.phi)
            ),
            0,
            self.upper_cutoff,  # factor np.exp(1.j * self.phi) comes from integration measure
        )

        # compute imaginary part by using integration routine
        right_segment_cont_imag, _ = integrate.quad(
            lambda x: np.imag(
                np.exp(1.0j * self.phi) * distr(t, x, self.beta, self.phi)
            ),
            0,
            self.upper_cutoff,  # factor np.exp(1.j * self.phi) comes from integration measure
        )

        return right_segment_cont_real + 1.0j * right_segment_cont_imag

    def discrete_integral(self, t):
        """
        Computes the discrete approximation to the frequency integral at fixed time t
        Parameters:
            - t (float): time point argument
        Returns:
        np.complex_: Discrete approximation result to frequency integral at fixed time t
        """

        # compute the discrete approximation to the frequency integral at fixed time t
        right_segment = np.sum(
            [
                distr(
                    t,
                    np.exp(self.h * k - np.exp(-self.h * k)) * np.exp(1.0j * self.phi),
                    self.beta,
                    phi=0,
                )  # set phi=0, because the argument to the phase is already included here and should not be added again in the function distr()
                * self.h#the following lines are from the integration measure
                * np.exp(1.0j * self.phi)
                * (1 + np.exp(-self.h * k))
                * np.exp(self.h * k - np.exp(-self.h * k))
                * spec_dens(omega = np.exp(self.h * k - np.exp(-self.h * k)) * np.exp(1.0j * self.phi))#spectral density evaluated at complex frequency
                for k in range(-self.n, self.m + 1)
            ]
        )

        return right_segment

    def abs_error(self, t):
        """
        Compute the absolute deviation between the continous intgeral and the discrete approximation at fixed time t
        Parameters:
        - t (float): time argument

        Returns:
        - float: Absolute devation between continous integration result and discrete approximation at time t
        """

        cont_val = self.cont_integral(t)
        discr_val = self.discrete_integral(t)

        return abs(cont_val - discr_val)

    def time_integrate(self, integral_vals):
        """
        Compute the time-integrated value based on a time-series:
        - np.array(float): time series to be integrated

        Returns:
        - float: time-integrated value
        """
        time_integrated_value = self.delta_t * np.sum(integral_vals)
        return time_integrated_value

    def error_time_integrated(self, cont_integral=None, discr_integral = None, error_type="rel"):
        """
        Compute the time-integrated deviation between the continous integral and the discrete approximation. Time-integration is performed on discrete time grid "times"
        Parameters:
        - np.array(float) [optional]: array containing continuous integration result for all points on time grid
        - error_type (string): Choose between 'rel' (default) and 'abs' for relative or absolute error, respectively.

        Returns:
        - float: time-integrated error between continous integration result and discrete approximation
        """

        # if no values for discrete integral are specified, compute them here
        if discr_integral is None:
            discr_integral = np.array([self.discrete_integral(t) for t in self.times])

        # if no values for continuous integral are specified, compute them here
        if cont_integral is None:
            cont_integral = np.array([self.cont_integral(t) for t in self.times])

        error_time_integrated = self.time_integrate(
            abs(cont_integral - discr_integral)
        )  # absolute time-integrated error

        if error_type == "rel":  # relative error defined such that it is always \leq 1
            norm = self.time_integrate(abs(cont_integral) + abs(discr_integral))
            error_time_integrated *= (
                1.0 / norm
            )  # turn into relative time-integrated error

        return error_time_integrated


class RtKernel:
    def __init__(self, N_max, delta_t, beta, upper_cutoff, m, n, eps, h, phi=np.pi / 4):
        """
        Parameters:
        - N_max (int): nbr. of time steps up to final time
        - delta_t (float): time discretization step
        - beta (float): inverse temperature
        - upper_cutoff (float): maximal energy considered
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - eps (float): error for interpolvative decomposition (ID) and singular value decomposition (SVD)
        - phi (float): rotation angle in complex plane
        - h (float): Discretization parameter
        """
        # initialize frequency grid

        self.h = h

        self.fine_grid = np.array(
            [np.exp(self.h * k - np.exp(-self.h * k)) for k in range(-n, m + 1)]
        )  # create fine grid according to superexponential formula

        # initialize time grid
        self.times = set_time_grid(N_max, delta_t)

        # create kernel matrix K on fine grid
        self.K = np.array(
            [
                [
                    distr(
                        t,
                        np.exp(self.h * k - np.exp(-self.h * k)) * np.exp(1.0j * phi),
                        beta,
                    )
                    * self.h
                    * np.exp(1.0j * phi)
                    * (1 + np.exp(-self.h * k))
                    * np.exp(self.h * k - np.exp(-self.h * k))
                    for k in range(-n, m + 1)
                ]
                for t in self.times
            ]
        )

        #__________perform SVD on kernel K and count number of singular values above error threshold____________
        (
            self.num_singular_values_above_threshold,
            self.singular_values,
        ) = svd_check_singular_values(self.K, eps)

        # perform ID on K
        self.ID_rank, self.idx, self.proj = sli.interp_decomp(
            self.K, eps
        )  # Comment: The fast version of this algorithm from the scipy library uses random sampling and may not give completely identical results for every run. See documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html". Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank, see documentation (access: 6. Dec. 2023)

        # compute coarse grid
        self.coarse_grid = np.array(self.fine_grid[self.idx[: self.ID_rank]])



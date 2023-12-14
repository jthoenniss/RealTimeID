import numpy as np
from scipy import integrate
import scipy.linalg.interpolative as sli
import math


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

    return 1.0


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
        lambda x: np.real(
            np.exp(1.0j * phi)
            * distr(t, x, beta, phi)
            * spec_dens(omega=x * np.exp(1.0j * phi))
        ),
        0,
        upper_cutoff,  # factor np.exp(1.j * phi) comes from integration measure
    )

    # compute imaginary part by using integration routine
    right_segment_cont_imag, _ = integrate.quad(
        lambda x: np.imag(
            np.exp(1.0j * phi)
            * distr(t, x, beta, phi)
            * spec_dens(omega=x * np.exp(1.0j * phi))
        ),
        0,
        upper_cutoff,  # factor np.exp(1.j * self.phi) comes from integration measure
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
        limits = np.arange(lower_limit, upper_limit)
        point_density = np.array(
            [np.sum((grid >= 10.0**a) & (grid < 10.0 ** (a + 1))) for a in limits]
        )
        point_density_grid = np.array(
            [(10.0**a + 10.0 ** (a + 1)) / 2.0 for a in limits]
        )

    else:
        raise ValueError(
            "Invalid interval spacing parameter specified. Use 'lin' or 'log'."
        )

    return point_density, point_density_grid


class DiscrError:
    def __init__(
        self,
        m,
        n,
        N_max,
        delta_t,
        beta,
        upper_cutoff,
        phi=np.pi / 4,
        times=None,
        h=None,
    ):
        self.m = m
        self.n = n
        self.N_max = N_max
        self.delta_t = delta_t
        self.beta = beta
        self.upper_cutoff = upper_cutoff
        self.phi = phi
        self.error = None
        self.times = (
            set_time_grid(self.N_max, self.delta_t) if times is None else times
        )
        self.h = (np.log(upper_cutoff) / self.m) if h is None else h

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
                    np.exp(self.h * k - np.exp(-self.h * k)),
                    self.beta,
                    phi=self.phi,
                )  # set phi=0, because the argument to the phase is already included here and should not be added again in the function distr()
                * self.h  # the following lines are from the integration measure
                * np.exp(1.0j * self.phi)
                * (1 + np.exp(-self.h * k))
                * np.exp(self.h * k - np.exp(-self.h * k))
                * spec_dens(
                    omega=np.exp(self.h * k - np.exp(-self.h * k))
                    * np.exp(1.0j * self.phi)
                )  # spectral density evaluated at complex frequency
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

    def time_integrate(self, time_series):
        """
        Compute the time-integrated value based on a time-series:
        - times_series(np.array(float)): time series to be integrated

        Returns:
        - float: time-integrated value
        """
        time_integrated_value = self.delta_t * np.sum(time_series)
        return time_integrated_value

    def error_time_integrated(
        self,
        time_series_exact=None,
        time_series_approx=None,
        error_type="rel",
        set_state_variable=False,
    ):
        """
        Compute the time-integrated deviation between two time series, e.g. between a continous frequency integral and the discrete approximation. Time-integration is performed on discrete time grid "times"
        Parameters:
        - times_series_exact (np.array(float)) [optional]: array containing exact time series for all points on time grid. If not specified, compute continous-frequency integral below.
        - times_series_approx(np.array(float)) [optional]: array containing approximate time series for all points on time grid. If not specified, compute discrete-frequency integral below.
        - error_type (string): Choose between 'rel' (default) and 'abs' for relative or absolute error, respectively.
        - set_state_variable (bool): If True, the error between the two time series is stored as state variable.
        Returns:
        - float: time-integrated error between exact and approximate time series
        """

        # if no values for discrete integral are specified, compute them here
        if time_series_approx is None:
            time_series_approx = np.array(
                [self.discrete_integral(t) for t in self.times]
            )

        # if no values for continuous integral are specified, compute them here
        if time_series_exact is None:
            time_series_exact = np.array([self.cont_integral(t) for t in self.times])

        error_time_integrated = self.time_integrate(
            abs(time_series_exact - time_series_approx)
        )  # absolute time-integrated error

        if error_type == "rel":  # relative error defined such that it is always \leq 1
            norm = self.time_integrate(abs(time_series_exact) + abs(time_series_approx))
            error_time_integrated *= (
                1.0 / norm
            )  # turn into relative time-integrated error

        if set_state_variable:  # set state variable
            self.error = error_time_integrated

        return error_time_integrated

    def optimize(self, time_series_exact=None):
        """
        Optimize the number of modes (m and n) to balance accuracy and computational cost.

        Parameters:
            time_series_exact (numpy.ndarray): Array containing the exact values of the continuous integral. If not provided, it will be computed below

        Returns:
            None
        """
        # Compute continuous integral if not provided
        cont_integral = (
            time_series_exact
            if time_series_exact is not None
            else np.array([self.cont_integral(t) for t in self.times])
        )

        discr_integral_init = np.array(
            [self.discrete_integral(t) for t in self.times]
        )  # discrete frequency integral approximation in the limit of large m and n

        err = self.error_time_integrated(
            time_series_exact=cont_integral, time_series_approx=discr_integral_init
        )  # compute error between discrete integral and continuous integral

        m_init, n_init = self.m, self.n
        m_final, n_final = m_init, n_init

        # Optimization for m while n is fixed to n_init
        for _ in range(m_init):
            self.m -= 1

            # Check if the error to the discrete integral in the large m,n limit differs by more than epsilon
            rel_val_diff = self.error_time_integrated(
                time_series_exact=discr_integral_init,
                time_series_approx=None,
                error_type="rel",
            )  # by setting time_series_approx to None, the discrete approximation is computed implicitly with self.m and self.n
            if (
                rel_val_diff > 0.1 * err
            ):  # reduce m until the error from finite m becomes on the order of 10% of the discretization error due to finite h, and thus remains othe subdominant error
                # If the value differs, halt iteration through m
                m_final = self.m + 1
                # for the optimization of n, temporarily reset self.m to m_init
                self.m = m_init
                break

        # Optimization for n while m is fixed to m_init
        for _ in range(n_init):
            self.n -= 1

            # Check if the error to the discrete integral in the large m,n limit differs by more than epsilon
            rel_val_diff = self.error_time_integrated(
                time_series_exact=discr_integral_init,
                time_series_approx=None,
                error_type="rel",
            )  # by setting time_series_approx to None, the discrete approximation is computed implicitly with self.m and self.n
            if (
                rel_val_diff > 0.1 * err
            ):  # reduce m until the error from finite m becomes on the order of 10% of the discretization error due to finite h, and thus remains othe subdominant error
                # If the value differs, halt iteration through n
                n_final = self.n + 1
                break

        # Update m and n with the optimized values
        self.m, self.n = m_final, n_final

        # compute the relative time integrated error w.r.t. continous integral
        self.rel_error_to_cont = self.error_time_integrated(
            time_series_exact=cont_integral,
            time_series_approx=None,
            error_type="rel",
            set_state_variable=True,
        )  # by setting time_series_approx to None, the discrete approximation is computed implicitly with self.m and self.n. Set state variable.

        return self


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

        # __________perform SVD on kernel K and count number of singular values above error threshold____________
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

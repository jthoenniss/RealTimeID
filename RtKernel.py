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


def generate_composite_chebyshev_grid_dyadic(M_intervals, m_chebyshev, cutoff):
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

    return cutoff * np.array(cheb_points)


def set_time_grid(t_max, delta_t):
    """
    Initializes the dicrete-time grid for fixed final time and time step
    Parameters:
    - t_max (float): maximal time
    - delta_t (float): time step

    Returns:
    np.array(): array containing the time points
    """
    return np.arange(delta_t, t_max + delta_t, delta_t)


class DiscrError:
    def __init__(self, m, n, t_max, delta_t, beta, cutoff, phi=np.pi / 4):
        self.m = m
        self.n = n
        self.t_max = t_max
        self.delta_t = delta_t
        self.beta = beta
        self.cutoff = cutoff
        self.phi = phi

    def cont_integral(self, t):
        """
        Perform frequency integral in continuous-frequency limit in interval [0,cutoff], at fixed time t
        Parameters:
        - t (float): time argument
        - beta (float): inverse temperature
        - phi (float): rotations angle in complex plane
        - cutoff (float): energy cutoff up to which kernel is integrated

        Returns:
        - (np.complex_): Result of integration in interval [0,cutoff]
        """
        # compute real part by using integration routine
        right_segment_cont_real, _ = integrate.quad(
            lambda x: np.real(
                np.exp(1.0j * self.phi) * distr(t, x, self.beta, self.phi)
            ),
            0,
            self.cutoff,  # factor np.exp(1.j * self.phi) comes from integration measure
        )

        # compute imaginary part by using integration routine
        right_segment_cont_imag, _ = integrate.quad(
            lambda x: np.imag(
                np.exp(1.0j * self.phi) * distr(t, x, self.beta, self.phi)
            ),
            0,
            self.cutoff,  # factor np.exp(1.j * self.phi) comes from integration measure
        )

        return right_segment_cont_real + 1.0j * right_segment_cont_imag

    def discrete_integral(self, t):
        """
        Computes the discrete approximation to the frequency intergal at fixed time t
        Parameters:
            - m (int): number of discretization intervals for omega > 1
            - n (int): number of discretization intervals for omega < 1
            - t (float): time point argument
            - beta (float): inverse temperature
            - cutoff (float): maximal energy considered
            - phi (float): rotation angle in complex plane

        Returns:
        np.complex_: Discrete approximation result to frequency integral at fixed time t
        """
        h = (
            np.log(self.cutoff) / self.m
        )  # compute the discretization parameter h based on cutoff and number of frequency intervals

        # compute the discrete approximation to the frequency integral at fixed time t
        right_segment = np.sum(
            [
                distr(
                    t,
                    np.exp(h * k - np.exp(-h * k)) * np.exp(1.0j * self.phi),
                    self.beta,
                    phi=0,
                )  # set phi=0, because the argument to the phase is already included here and should not be added again in the function distr()
                * h
                * np.exp(1.0j * self.phi)
                * (1 + np.exp(-h * k))
                * np.exp(h * k - np.exp(-h * k))
                for k in range(-self.n, self.m + 1)
            ]
        )

        return right_segment

    def abs_error(self, t):
        """
        Compute the absolute deviation between the continous intgeral and the discrete approximation at fixed time t
        Paramters:
        - t (float): time argument

        Returns:
        np.complex_: Absolute devation between continous integration result and discrete approximation at time t
        """
        val_cont = self.cont_integral(t)
        val_disc = self.discrete_integral(t)
        return abs(val_cont - val_disc)

    def abs_error_time_integrated(self):
        """
        Compute the absolute time-integrated deviation between the continous intgeral and the discrete approximation. Time-integration is performed on discrete time grid "times"
        Paramters:
        - None

        Returns:
        np.complex_: Absolute time-integrated devation between continous integration result and discrete approximation
        """
        times = set_time_grid(self.t_max, self.delta_t)
        abs_error_time_integrated = self.delta_t * np.sum(
            [self.abs_error(t) for t in times]
        )
        return abs_error_time_integrated


class RtKernel:
    def __init__(self, t_max, delta_t, beta, cutoff, m, n, eps, phi=np.pi / 4):
        """
        Parameters:
        - t_max (float): maximal time up to which dynamics is computed
        - delta_t (float): time discretization step
        - beta (float): inverse temperature
        - cutoff (float): maximal energy considered
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - eps (float): error for interpolvative decomposition (ID) and singular value decomposition (SVD)
        - phi (float): rotation angle in complex plane
        """
        # initialize frequency grid
        self.h = (
            np.log(cutoff) / m
        )  # choose discretization parameter h such that the highest frequency is the cutoff
        self.fine_grid = np.array(
            [np.exp(self.h * k - np.exp(-self.h * k)) for k in range(-n, m + 1)]
        )  # create fine grid according to superexponential formula

        # initialize time grid
        self.times = set_time_grid(t_max, delta_t)

        # create kernel matrix K on fine grid
        K = np.array(
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

        # perform SVD on kernel K and count number of singular values above error threshold
        (
            self.num_singular_values_above_threshold,
            self.singular_values,
        ) = svd_check_singular_values(K, eps)

        # perform ID on K
        self.ID_rank, self.idx, self.proj = sli.interp_decomp(
            K, eps
        )  # Comment: The fast version of this algorithm from the scipy library uses random sampling and may not give completely identical results for every run. See documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html" (access: 30. Nov. 2023)

        self.P = np.hstack([np.eye(self.ID_rank), self.proj])[
            :, np.argsort(self.idx)
        ]  # projection matrix

        # compute coarse grid
        self.coarse_grid = np.array(self.fine_grid[self.idx[: self.ID_rank]])

  
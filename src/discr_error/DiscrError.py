import numpy as np
from src.utils import common_funcs as cf


class DiscrError:
    def __init__(
        self,
        m: int,
        n: int,
        N_max: int,
        delta_t: float,
        beta: float,
        upper_cutoff: float,
        times: np.ndarray,
        h: float,
        phi: float,
    ):
        """
        Parameters:
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - N_max (int): nbr. of time steps up to final time
        - delta_t (float): time discretization step
        - beta (float): inverse temperature
        - upper_cutoff (float): maximal energy considered in continous integration
        - times (numpy.ndarray): Array containing the points on the time grid
        - h (float): Discretization parameter
        - phi (float): rotation angle in complex plane

        Note: Either "N_max" and "delta_t" OR "times" needs to be specified to define the time grid. If all is specified, the argument "times" is used as time grid.
        """

        # Determine the time grid based on provided information or generate one
        self.times = times if times is not None else cf.set_time_grid(N_max, delta_t)
        self.N_max = len(self.times)
        self.delta_t = np.diff(self.times).mean() if len(self.times) > 1 else 0.0


        self.m = m
        self.n = n
        self.beta = beta
        self.upper_cutoff = upper_cutoff
        self.h = h
        self.phi = phi
        self.eps = None  # to store the eps w.r.t.to exact result

    def cont_integral(self, t):
        """
        Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff], at fixed time t
        Parameters:
        - t (float): time argument
        Returns:
        - (np.complex_): Result of integration in interval [0,upper_cutoff]
        """
        return cf.cont_integral(t, self.beta, self.upper_cutoff, phi=self.phi)

    def discrete_integral(self, t):
        """
        Computes the discrete approximation to the frequency integral at fixed time t
        Parameters:
            - t (float/np.ndarray): time point argument as single float or array
        Returns:
        np.complex_ (or in np.ndrray): Discrete approximation result to frequency integral at fixed time t
        """
        t_array = np.asarray(t)[:, np.newaxis]#if t is a scalar, convert to numpy array and reshape to enable broadcasting

        k_values = np.arange(-self.n, self.m + 1)
        exp_k_values = np.exp(
            -self.h * k_values
        )  # precompute as it is needed repeatedly
        fine_grid = np.exp(
            self.h * k_values - exp_k_values
        )  # exponential frequency grid with as many points as 'k_values' has entries
        fine_grid_complex = fine_grid * np.exp(
            1.0j * self.phi
        )  # fine grid rotated into complex plane by angle phi.
        K = cf.distr(
            t_array, fine_grid_complex, self.beta
        )  # kernel defined by Fermi-distribution cf.distr and the time-dependent Fourier factor e^{i\phi t}
        K *= (
            self.h
            * (1 + exp_k_values)
            * fine_grid_complex
            * cf.spec_dens_array(
                omega_array=fine_grid_complex
            )  # spectral density evaluated at complex frequency
        )

        # Sum over the frequency axis
        right_segment = np.sum(K, axis=1)

        # Return the original shape (float or 1D array)
        return right_segment if t_array.ndim > 1 else right_segment[0]

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
        store_eps=False,
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
            time_series_approx = self.discrete_integral(self.times) 

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

        if store_eps:  # set state variable
            self.eps = error_time_integrated

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

        discr_integral_init = self.discrete_integral(self.times) # discrete frequency integral approximation in the limit of large m and n

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
            store_eps=True,
        )  # by setting time_series_approx to None, the discrete approximation is computed implicitly with self.m and self.n. Set state variable.

        return self

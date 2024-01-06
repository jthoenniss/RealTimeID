import numpy as np
from src.utils import common_funcs as cf
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.kernel_params.parameter_validator import ParameterValidator


class DiscrError(KernelMatrix):

    """
    Represents a class designed to evaluate and quantify the discretization error encountered when approximating a continuous frequency integral. This error is assessed by comparing two methodologies:
    (i) an approximation using a discretized path in the complex plane, and
    (ii) a computation using a continuous integration routine along the same path.
    The class provides functionality to calculate both the discrete and continuous integrals, compute their deviation, and optimize the discretization parameters for balanced accuracy and computational efficiency.

    Attributes:
        discrete_integral_init (np.ndarray): Initial discrete integral approximation.
        cont_integral_init (np.ndarray): Initial continuous integral, set upon class instantiation if provided.
        eps (float): The time-integrated error relative to the continuous integral result.
        upper_cutoff (float): Upper frequency cutoff for continuous integration.
        times (np.ndarray): Array of time points used in the integrals.

    Methods:
        cont_integral(): Computes the continuous frequency integral in the interval [0, upper_cutoff].
        discrete_integral(): Calculates the discrete approximation of the frequency integral at predefined time points.
        time_integrate(time_series): Integrates a given time series over the time grid.
        error_time_integrated(time_series_exact, time_series_approx): Computes the time-integrated deviation between two time series.
        optimize(): Optimizes the number of modes (m and n) for a balance between accuracy and computational cost.
        get_params(): Returns a dictionary of parameters associated with the class instance.
    """

    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        upper_cutoff: float,
        N_max: int,
        delta_t: float,
        h: float,
        phi: float,
        spec_dens: callable,
        cont_integral_init: np.ndarray = None
    ):
        """
        Parameters:
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - beta (float): inverse temperature
        - upper_cutoff (float): maximal energy considered in continous integration
        - N_max (int): number of points on time grid
        - delta_t (float): time step
        - h (float): Discretization parameter
        - phi (float): rotation angle in complex plane
        - spec_dens (callable): Single-parameter function for the spectral density
        - cont_integral_init (np.ndarray, optional): Array containing the continuous-time integral at all all points of the time grid.
        """
       
        super().__init__(
            m=m,
            n=n,
            beta=beta,
            N_max=N_max,
            delta_t=delta_t,
            h=h,
            phi=phi,
            spec_dens=spec_dens,
        )
     
        # compute discrete integral approximation
        self.discrete_integral_init = self.discrete_integral()

        # compute continous integral
        ParameterValidator.validate_upper_cutoff(upper_cutoff)
        self.upper_cutoff = (
            upper_cutoff  # upper frequency cutoff for continuous integration
        )

        self.cont_integral_init = (
            self.cont_integral() if cont_integral_init is None else cont_integral_init
        )
        
        # compute error between discrete and continuous integral
        self.eps = (
            self.error_time_integrated()
        )  # eps w.r.t. to continuous integral result

    def cont_integral(self):
        """
        Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff]

        Returns:
        - (np.complex_): Result of integration in interval [0,upper_cutoff]
        """
     
        return np.array(
            [
                cf.cont_integral(t = t, beta = self.beta, upper_cutoff= self.upper_cutoff, spec_dens = self.spec_dens, phi = self.phi)
                for t in self.times
            ]
        )

    def discrete_integral(
        self, kernel: np.ndarray = None, spec_dens_array_cmplx: np.ndarray = None
    ) -> np.ndarray:
        """
        Computes the discrete approximation to the frequency integral at the times defined on the time grid

        Parameters:
        - kernel (np.ndarray, optional): Kernel matrix, where different rows correspond to different time steps, and different columns correspond to different frequencies
        - spec_dens_array_cmplx (np.ndarray, optional): Array of spectral density values at the frequency points in the complex plane

        Returns:
        - np.ndarray: Discrete approximation result to frequency integral at times on time grid
        """

        # Use provided or default kernel and spec_dens_array
        kernel_eff = self.kernel if kernel is None else kernel
        spec_dens_array_eff_cmplx = (
            self.spec_dens_array_cmplx if spec_dens_array_cmplx is None else spec_dens_array_cmplx
        )

        if not isinstance(kernel_eff, np.ndarray):
            raise TypeError(
                f"'kernel' must be of type np.ndarray. Found {type(kernel_eff).__name__}"
            )
        
        if not isinstance(spec_dens_array_eff_cmplx, np.ndarray):
            raise TypeError(
                f"'spec_dens_array' must be of type np.ndarray. Found {type(spec_dens_array_eff_cmplx).__name__}"
            )
        if kernel_eff.shape[1] != len(spec_dens_array_eff_cmplx):
            raise RuntimeError(
                f"Frequency dimension of 'kernel' must match length of 'spec_dens_array'. Respective values found: {kernel_eff.shape[1]}, {len(spec_dens_array_cmplx)}"
            )

        # Sum over the frequency axis
        right_segment = kernel_eff @ spec_dens_array_eff_cmplx

        return right_segment

    def time_integrate(self, time_series):
        """
        Compute the time-integrated value based on a time-series:
        - times_series(np.array(float)): time series to be integrated

        Returns:
        - float: time-integrated value
        """
        time_integrated_value = self.delta_t * np.sum(time_series)
        return time_integrated_value

    def error_time_integrated(self, time_series_exact=None, time_series_approx=None):
        """
        Compute the time-integrated deviation between two time series, e.g. between a continous frequency integral and the discrete approximation. Time-integration is performed on discrete time grid "times"
        Parameters:
        - times_series_exact (np.array(float)) [optional]: array containing exact time series for all points on time grid. If not specified, compute continous-frequency integral below.
        - times_series_approx(np.array(float)) [optional]: array containing approximate time series for all points on time grid. If not specified, compute discrete-frequency integral below.
        Returns:
        - float: time-integrated error between exact and approximate time series
        """

        # if no values for discrete integral are specified, take attribute variable for time_series_approx
        time_series_approx = (
            self.discrete_integral_init
            if time_series_approx is None
            else time_series_approx
        )
        # if no values for continuous integral are specified,take attribute variable for time_series_exact
        time_series_exact = (
            self.cont_integral_init if time_series_exact is None else time_series_exact
        )

        # compute absolute time integrated error
        abs_error_time_integrated = self.time_integrate(
            abs(time_series_exact - time_series_approx)
        )
        # compute norm
        norm = self.time_integrate(abs(time_series_exact) + abs(time_series_approx))
        # compute relative error by dividing by norm
        rel_error_time_integrated = abs_error_time_integrated / norm

        return rel_error_time_integrated

    def optimize(self):
        """
        Optimize the number of modes (m and n) to balance accuracy and computational cost.
        """
        nbr_freqs = len(self.fine_grid)
        # search for number of points one can spare in m and n without making an error that dominate the discretization error w.r.t. to continuous integration
        m_count_final = self._optimize_mode_count(
            self.m, lambda mc: [0, nbr_freqs - mc]
        )
        n_count_final = self._optimize_mode_count(self.n, lambda nc: [nc, nbr_freqs])

        # update attribute variables for number of discretization points
        self.m -= m_count_final
        self.n -= n_count_final

        # update kernel
        self._update_kernel_and_grids()

        return self

    def _optimize_mode_count(self, max_count, interval_idcs):
        """
        Helper method to optimize mode count (either m or n), such that the error introduced by cutting large or small frequencies
        is at most 10% of the discretization error.

        Parameters:
        max_count (int): maximal number of frequencies by which the interval can be shrinked
        interval_icds (array): array containing the lower and upper bound of the frequency interval considered

        Returns:
        - int: number of frequency points dropped without making a error larger than 10% of the discretization error
        """
        for count in range(1, max_count):
            # Compute kernel and spec_dens_array on reduced frequency grid
            kernel_reduced = self.kernel[:, interval_idcs[0] : interval_idcs[1]]
            spec_dens_array_cmplx_reduced = self.spec_dens_array_cmplx[
                interval_idcs[0] : interval_idcs[1]
            ]
            # compute the corresponding discrete-frequency approximation
            time_series_approx = self.discrete_integral(
                kernel=kernel_reduced, spec_dens_array_cmplx=spec_dens_array_cmplx_reduced
            )
            # compute relative error between large m/n discrete integral approximation and approximation with current m and n
            rel_val_diff = self.error_time_integrated(
                time_series_exact=self.discrete_integral_init,
                time_series_approx=time_series_approx,
            )

            if rel_val_diff > 0.1 * self.eps:
                return count - 1  # Found the optimal count
        return (
            max_count - 1
        )  # In case no optimal count is found, return the last valid count

    def _update_kernel_and_grids(self):
        """
        Update the kernel and dependent attributes after changes to m or n.
        For this, run _update_kernel from base class and additionally recompute the discrete integral, as well as the error 'eps'.
        """
        #update kernel and grids
        super()._initialize_kernel_and_grids()
        # update discrete_integral_init
        self.discrete_integral_init = self.discrete_integral()
        # update error between continous and discrete integral
        self.eps = self.error_time_integrated()

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_dict = super().get_params()

        param_dict["eps"] = getattr(
            self, "eps"
        )  # add parameters for eps which does not exist in base class KernelMatrix.

        return param_dict

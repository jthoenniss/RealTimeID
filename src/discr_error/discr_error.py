import numpy as np
from src.utils import common_funcs as cf
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.kernel_params.kernel_params import KernelParams
import scipy.special as sp # for lambertw

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

        # compute discrete integral
        self.discrete_integral_init = self.discrete_integral()

        # compute continous integral
        KernelParams.validate_upper_cutoff(upper_cutoff)
        self.upper_cutoff = upper_cutoff

        self.cont_integral_init = (
            self.cont_integral() if cont_integral_init is None else cont_integral_init
        )

        # compute time-integrated error between discrete and continuous integral
        self.eps = self.error_time_integrated()

    def cont_integral(self):
        """
        Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff]

        Returns:
        - (np.complex_): Result of integration in interval [0,upper_cutoff]
        """

        return cf.cont_integral(
            t=self.times,
            beta=self.beta,
            upper_cutoff=self.upper_cutoff,
            spec_dens=self.spec_dens,
            phi=self.phi,
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
            self.spec_dens_array_cmplx
            if spec_dens_array_cmplx is None
            else spec_dens_array_cmplx
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

    def optimize(self, update_params: KernelParams = None, rel_error_diff: float  = 1.e-16) -> "DiscrError":
        """
        Optimize the number of modes (m and n) to balance accuracy and computational cost.
        Parameters:
        - update_params (KernelParams, optional): An instance of KernelParams that holds the parameter set.
        - rel_error_diff (float, optional): Threshold for the relative difference between the old and new error
        """
        nbr_freqs = len(self.fine_grid)

        #temporarily store current variables to compare to optimizd variables
        freq_limits_prev =  (self.fine_grid[0], self.fine_grid[-1])
        eps_prev = self.eps

        # search for number of points one can spare in m and n without making an error that dominate the discretization error w.r.t. to continuous integration
        m_count_final = self._optimize_mode_count(self.m, lambda mc: [0, nbr_freqs - mc], rel_error_diff)
        n_count_final = self._optimize_mode_count(self.n, lambda nc: [nc, nbr_freqs], rel_error_diff)

        # update m,n, kernel, grids for frequency, and spectr. density, discrete integral, and eps
        self._update_reduced_kernel_and_grids(m_count_final, n_count_final)

        if update_params is not None:  
            self._update_external_params(update_params)

        #uncomment to print optimization results
        self._print_optimization_results(freq_limits_prev, eps_prev)

        return self
    



    def _update_external_params(self, update_params: KernelParams) -> None:
        """
        Update the external KernelParams object and determine finite limits for discrete integral,
        such that one does not need to seach from large grid everytime
        Attention: When going from large to small errors, the limits determined in a previous calculation might not be sufficient in order to obtain a smaller error.

        Parameters:
        - update_params (KernelParams): An instance of KernelParams that holds the parameter set.

        Returns:
        - None
        """
        # update parameters in external KernelParams object
        update_params.update_parameters({"m": self.m,"n": self.n})

        # update discrete cutoffs in external KernelParams object
        w_min, w_max = self.fine_grid[0], self.fine_grid[-1]
        lower_cutoff_argument_discrete = - np.log (w_min) - np.real(sp.lambertw(1/w_min))#Choose cutoff, such that exp(-cutoff - exp(cutoff)) = w_min
        upper_cutoff_argument_discrete = np.log (w_max) + np.real(sp.lambertw(1/w_max))#Choose cutoff, such that exp(cutoff - exp(-cutoff)) = w_max
        update_params.set_discrete_cutoffs(lower_cutoff_argument_discrete=lower_cutoff_argument_discrete, upper_cutoff_argument_discrete=upper_cutoff_argument_discrete)


    def _print_optimization_results(self,freq_limits_prev, eps_prev):
            """
            Helper method to print the results of the optimization
            """
            print(f"For h={np.format_float_scientific(self.h, precision=3)}, beta={np.round(self.beta, 2)}, T_max={np.round(self.N_max*self.delta_t, 2)}: Changed frequency limits from [{np.format_float_scientific(freq_limits_prev[0], precision=2)}, {np.format_float_scientific(freq_limits_prev[1], precision=2)}] to [{np.format_float_scientific(self.fine_grid[0], precision=2)}, {np.format_float_scientific(self.fine_grid[-1], precision=2)}].")
            print(f"Relative error previously: {np.format_float_scientific(eps_prev, precision=2)}, now: {np.format_float_scientific(self.eps, precision=2)}.")


    def _optimize_mode_count(self, max_count, interval_idcs, rel_error_diff):
        """
        Helper method to optimize mode count (either m or n), such that the relative difference between the old and new error
        is at most 10% of the discretization error.

        Parameters:
        - max_count (int): maximal number of frequencies by which the interval can be shrinked
        - interval_icds (array): array containing the lower and upper bound of the frequency interval considered
        - rel_error_diff (float): threshold for the relative difference between the old and new error


        Returns:
        - int: number of frequency points dropped without making a error larger than 10% of the discretization error
        """
        for count in range(1, max_count):
           
            lower_idx, upper_idx = interval_idcs(count)[0], interval_idcs(count)[1]
            
            # Compute the error between the old and new discrete integral
            eps_reduced = self._get_reduced_kernel_and_error(lower_idx, upper_idx)[
                "eps_reduced"
            ]

            if eps_reduced / self.eps > rel_error_diff:
                return count - 1  # Found the optimal count
      
                    
        # In case no optimal count is found, return the last valid count
        return max_count - 1 

    def _get_reduced_kernel_and_error(self, lower_idx: int, upper_idx: int) -> dict:
        """
        Helper method to extract the reduced kernel and grids for a given frequency interval.
        Based on this, a new discrete integral is computed and the respective relative error between the old and new discrete integral with the continous integral is computed.

        Parameters:
        - lower_idx (int): lower index of frequency interval
        - upper_idx (int): upper index of frequency interval

        Returns:
        - dict: dictionary containing the reduced kernel, reduced spec_dens_array_cmplx, reduced discrete integral, and reduced error
        """
        # compute reduced kernel and spec_dens_array_cmplx
        kernel_reduced = self.kernel[:, lower_idx:upper_idx]
        spec_dens_array_cmplx_reduced = self.spec_dens_array_cmplx[lower_idx:upper_idx]

        # compute the corresponding discrete-frequency approximation
        discrete_integral_reduced = self.discrete_integral(
            kernel=kernel_reduced, spec_dens_array_cmplx=spec_dens_array_cmplx_reduced
        )
        # compute relative error between the discrete integral approximation with current m and n and the discrete integral with previous m and n
        eps_reduced = self.error_time_integrated(
            time_series_exact=self.discrete_integral_init,
            time_series_approx=discrete_integral_reduced,
        )
   
        return {
            "kernel_reduced": kernel_reduced,
            "spec_dens_array_cmplx_reduced": spec_dens_array_cmplx_reduced,
            "discrete_integral_reduced": discrete_integral_reduced,
            "eps_reduced": eps_reduced,
        }

    def _update_reduced_kernel_and_grids(
        self, m_count: int, n_count: int
    ) -> None:
        """
        Helper method to update the kernel and grids based on the optimized number of discretization points for m and n

        Parameters:
        - m_count (int): final number of discretization points for m
        - n_count (int): final number of discretization points for n

        Returns:
        - None
        """

        nbr_freqs = len(self.fine_grid)

        self.m -= m_count
        self.n -= n_count

        # get reduced kernel, spec_dens_array_cmplx, discrete integral, and error
        new_kernel_and_grids = self._get_reduced_kernel_and_error(
            lower_idx=n_count, upper_idx=nbr_freqs - m_count
        )

        # update kernel, spec_dens_array_cmplx, discrete integral, and error
        self.kernel = new_kernel_and_grids["kernel_reduced"]
        self.spec_dens_array_cmplx = new_kernel_and_grids[
            "spec_dens_array_cmplx_reduced"
        ]
        self.discrete_integral_init = new_kernel_and_grids["discrete_integral_reduced"]

        # update error between discrete and continuos integral, both stored as attributes
        # (comment: don't use new_kernel_and_grids["eps_reduced"] here because that is the error between the new and the old DISCRETE integral
        # and not between the new discrete integral and the continuous integral)
        self.eps =  self.error_time_integrated()
   
        # update frequency grid and k values
        self.fine_grid = self.fine_grid[n_count : nbr_freqs - m_count]
        self.k_values = self.k_values[n_count : nbr_freqs - m_count]

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_dict = super().get_params()

        param_dict["eps"] = getattr(
            self, "eps"
        )  # add parameters for eps which does not exist in base class KernelMatrix.

        return param_dict

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
        """
    
        super().__init__(m = m, n = n, beta = beta, N_max= N_max, delta_t= delta_t, h = h, phi = phi)

        #compute discrete integral approximation
        self.discrete_integral_init = self.discrete_integral()

        #compute continous integral
        ParameterValidator.validate_upper_cutoff(upper_cutoff)
        self.upper_cutoff = upper_cutoff#upper frequency cutoff for continuous integration
        
        self.cont_integral_init = self.cont_integral() if cont_integral_init is None else cont_integral_init

        #compute error between discrete and continuous integral
        self.eps = self.error_time_integrated() #eps w.r.t. to continuous integral result


    def cont_integral(self):
        """
        Perform frequency integral in continuous-frequency limit in interval [0,upper_cutoff]
       
        Returns:
        - (np.complex_): Result of integration in interval [0,upper_cutoff]
        """
        return np.array([cf.cont_integral(t, self.beta, self.upper_cutoff, self.phi) for t in self.times])

    def discrete_integral(self, kernel = None):
        """
        Computes the discrete approximation to the frequency integral at the times defined on the time grid
       
        Returns:
        - np.ndarray: Discrete approximation result to frequency integral at times on time grid 
        """
        kernel_eff = self.kernel if kernel is None else kernel
        # Sum over the frequency axis
        right_segment = np.sum(kernel_eff, axis=1)

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

    def error_time_integrated(
        self,
        time_series_exact=None,
        time_series_approx=None
    ):
        """
        Compute the time-integrated deviation between two time series, e.g. between a continous frequency integral and the discrete approximation. Time-integration is performed on discrete time grid "times"
        Parameters:
        - times_series_exact (np.array(float)) [optional]: array containing exact time series for all points on time grid. If not specified, compute continous-frequency integral below.
        - times_series_approx(np.array(float)) [optional]: array containing approximate time series for all points on time grid. If not specified, compute discrete-frequency integral below.
        Returns:
        - float: time-integrated error between exact and approximate time series
        """

        # if no values for discrete integral are specified, take attribute variable for time_series_approx
        time_series_approx = self.discrete_integral_init if time_series_approx is None else time_series_approx
        # if no values for continuous integral are specified,take attribute variable for time_series_exact
        time_series_exact = self.cont_integral_init if time_series_exact is None else time_series_exact

        #compute absolute time integrated error
        abs_error_time_integrated = self.time_integrate(abs(time_series_exact - time_series_approx))  
        #compute norm
        norm = self.time_integrate(abs(time_series_exact) + abs(time_series_approx))
        #compute relative error by dividing by norm
        rel_error_time_integrated  = abs_error_time_integrated / norm

        return rel_error_time_integrated


    def optimize(self):
        """
        Optimize the number of modes (m and n) to balance accuracy and computational cost.
        """
        nbr_freqs = len(self.fine_grid)
        #search for number of points one can spare in m and n without making an error that dominate the discretization error w.r.t. to continuous integration
        m_count_final = self._optimize_mode_count(self.m, lambda mc: self.kernel[:, : nbr_freqs-mc])
        n_count_final = self._optimize_mode_count(self.n, lambda nc: self.kernel[:, nc:])

        #update attribute variables for number of discretization points
        self.m -= m_count_final
        self.n -= n_count_final
        
        #update kernel
        self._update_kernel()

        return self

    def _optimize_mode_count(self, max_count, kernel_slice_fn):
        """
        Helper method to optimize mode count (either m or n).
        """
        for count in range(1, max_count):
            time_series_approx = self.discrete_integral(kernel_slice_fn(count))
            #compute relative error between large m/n discrete integral approximation and approximation with current m and n
            rel_val_diff = self.error_time_integrated(time_series_exact=self.discrete_integral_init, time_series_approx=time_series_approx)

            if rel_val_diff > 0.1 * self.eps:
                return count - 1  # Found the optimal count
        return max_count - 1  # In case no optimal count is found, return the last valid count
        
      
    def _update_kernel(self):
        """
        Update the kernel and dependent attributes after changes to m or n.
        """
        super()._update_kernel()
        #update discrete_integral_init
        self.discrete_integral_init = self.discrete_integral()
        #update error between continous and discrete integral
        self.eps = self.error_time_integrated()

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_dict = super().get_params()

        param_dict["eps"] = getattr(self, "eps")#add parameters for eps which does not exist in base class KernelMatrix.

        return param_dict
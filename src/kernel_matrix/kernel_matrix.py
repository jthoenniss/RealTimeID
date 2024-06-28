import numpy as np
from src.utils import common_funcs as cf
from src.kernel_params.kernel_params import KernelParams
from typing import Tuple


class KernelMatrix:
    """
    A class that generates the kernel matrix associated with a Green's function.

    This class contains functions to compute a fine frequency grid, initialize the kernel matrix,
    and update the kernel matrix based on parameter changes.

    Parameters:
        - m (int): Number of discretization intervals for omega > 1/e.
        - n (int): Number of discretization intervals for omega < 1/e.
        - beta (float): Inverse temperature.
        - N_max (int): Number of points on the time grid.
        - delta_t (float): Time step.
        - h (float): Discretization parameter.
        - phi (float): Rotation angle in the complex plane.
        - spec_dens (callable): Spectral density as a function with one parameter
        - freq_parametrization (str): The parameterization of the frequency grid. Options are "simple_exp" and "fancy_exp".
            Simple exp: The grid is parametrized by omega_k = exp(h*k) for k in [-n, m].
            Fancy exp: The grid is parametrized by omega_k = exp(h*k - exp(-h*k)) for k in [-n, m].
        - only_positive_particle (bool): If True, only the positive frequencies for the particle component are included.
        - explicit_poles (np.ndarray): Array containing the explicit poles of the spectral density to be considered (default: empty array)
        - explicit_residues (np.ndarray): Array containing the residues of the spectral density to be considered (default: empty array)
        - additional_poles (np.ndarray): Array containing additional poles to be considered but not to be included into kernel matrix by default
        - additional_residues (np.ndarray): Array containing additional residues to be considered but not to be included into kernel matrix by default
    """

    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        N_max: int,
        delta_t: float,
        h: float,
        phi: float,
        spec_dens: callable,
        freq_parametrization: str,
        only_positive_particle: bool = False,
        explicit_poles: np.ndarray = np.empty(0),
        explicit_residues: np.ndarray = np.empty(0),
        additional_poles: np.ndarray = np.empty(0),
        additional_residues: np.ndarray = np.empty(0),
        **kwargs,
    ):
        # check if all parameters are valid
        KernelParams.validate_m_n(m, n)
        KernelParams.validate_beta(beta)
        KernelParams.validate_N_max_and_delta_t(N_max, delta_t)
        KernelParams.validate_h(h)
        KernelParams.validate_phi(phi)
        KernelParams.validate_freq_parametrization(freq_parametrization)

        for (
            kwarg
        ) in (
            kwargs
        ):  # ignore if an upper cutoff is specified as this is only relevant when computing continuous frequency integral as in DiscrKernel
            if kwarg == "upper_cutoff":
                pass
            else:
                raise ValueError(f"Invalid keyword argument: {kwarg}")

        # Store parameters
        self.m, self.n = m, n
        self.beta = beta
        self.N_max = N_max
        self.delta_t = delta_t
        self.h = h
        self.phi = phi
        self.spec_dens = spec_dens
        self.freq_parametrization = freq_parametrization
        self.only_positive_particle = only_positive_particle

        # Initialize kernel matrix and grids
        self._initialize_kernel_and_grids()

        # from explicit poles and residues, compute the kernel matrix
        self.explicit_poles, self.explicit_residues, self.kernel_explicit = (
            self._add_additional_poles_and_residues(
                explicit_poles, explicit_residues
            )
        )

        # from additional poles and residues, compute the additional kernel matrix
        self.additional_poles, self.additional_residues, self.kernel_additional = (
            self._add_additional_poles_and_residues(
                additional_poles, additional_residues
            )
        )

    def _initialize_kernel_and_grids(
        self,
    ) -> None:
        """
        (Re)compute the time grid, fine frequency grid, the kernel matrix and the vectorized spectrald density.
        Needed for initialization and after change of parameters.
        Parameters:
        - None
        Returns:
        - None
        """
        # set time grid
        self.times = cf.set_time_grid(N_max=self.N_max, delta_t=self.delta_t)
        # initialize frequency grid
        self.fine_grid, self.k_values, jacobian = cf.initialize_fine_grid(
            self.m, self.n, self.h, self.freq_parametrization
        )
        # full fine grid in complex plane
        self.fine_grid_complex = self.fine_grid * np.exp(1.0j * self.phi)
        if not self.only_positive_particle:  # if negative frequencies are included
            # add negative frequencies
            self.fine_grid_complex = np.concatenate(
                (-self.fine_grid_complex[::-1].conj(), self.fine_grid_complex)
            )

        # initialize matrix kernel
        self.kernel = self._initialize_kernel(jacobian=jacobian)

    def _initialize_kernel(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Creates the kernel matrix using the Fermi distribution function and spectral density.
        Parameters:
        - jacobian (np.ndarray): The Jacobian of the transformation from the measure: dw/dk.

        Returns:
        - np.ndarray: Kernel matrix.
        """
        times_arr = self.times[:, np.newaxis]  # enable broadcasting
        jacobian_cmplx_positive = jacobian * np.exp(
            1.0j * self.phi
        )  # adjust for complex contour

        if (
            self.only_positive_particle
        ):  # if only positive frequencies of the particle component are included
            # Kernel defined by Fermi distribution and spectral density
            # particle component
            K_particle = cf.dynamic_distr_particle(
                times_arr, self.fine_grid_complex, self.beta
            ) * self.spec_dens(self.fine_grid_complex)

            return K_particle * jacobian_cmplx_positive

        # add negative frequencies to jacobian
        jacobian_cmplx = np.concatenate(
            (jacobian_cmplx_positive[::-1].conj(), jacobian_cmplx_positive)
        )  # add negative frequencies

        # Kernel defined by Fermi distribution and spectral density
        # particle component
        K_particle = cf.dynamic_distr_particle(
            times_arr, self.fine_grid_complex, self.beta
        ) * self.spec_dens(self.fine_grid_complex)
        # hole component (negative sign in beta for hole distribution)
        K_hole = cf.dynamic_distr_particle(
            times_arr, self.fine_grid_complex, -self.beta
        ) * self.spec_dens(self.fine_grid_complex)

        # Combine particle and hole contributions by stacking them on top of each other
        K = np.vstack((K_particle, K_hole))

        K *= jacobian_cmplx  # multiply by Jacobian from measure. This is dw/dk. Multiply by exp(i*phi) to rotate in the complex plane

        return K

    def _add_additional_poles_and_residues(
        self, additional_poles: np.ndarray, additional_residues: np.ndarray
    ) -> None:
        """
        Compute the additional Kernel matrix if there are additional poles and residues given. If both are empty, the additional kernel matrix is an empty array.

        Parameters:
        - additional_poles (np.ndarray): array containing the additional poles of the spectral density to be considered.
        - additional_residues (np.ndarray): array containing the additional residues of the spectral density.
        Returns:
        - tuple: additional poles, additional residues, additional kernel matrix
        """

        # from additional poles and residues, compute the kernel matrix
        if len(additional_poles) != len(additional_residues):
            raise ValueError("The number of poles and residues must be the same")

        if self.only_positive_particle:  # only consider poles with positive real part
            additional_poles_eff = np.array(
                [pole for pole in additional_poles if pole.real > 0]
            )
            additional_residues_eff = np.array(
                [
                    res
                    for pole, res in zip(additional_poles, additional_residues)
                    if pole.real > 0
                ]
            )
        else:
            additional_poles_eff = additional_poles
            additional_residues_eff = additional_residues

        # particle component
        K_particle_add = (
            2.0j
            * np.pi
            * additional_residues_eff
            * cf.dynamic_distr_particle(
                self.times[:, np.newaxis], additional_poles_eff, self.beta
            )
        )
        if self.only_positive_particle:
            kernel_additional = K_particle_add
        else:
            # hole component (negative sign in beta for hole distribution)
            K_hole_add = (
                2.0j
                * np.pi
                * additional_residues_eff
                * cf.dynamic_distr_particle(
                    self.times[:, np.newaxis], additional_poles_eff, -self.beta
                )
            )
            # Combine particle and hole contributions by stacking them on top of each other
            kernel_additional = np.vstack((K_particle_add, K_hole_add))

        return additional_poles_eff, additional_residues_eff, kernel_additional

    def _full_kernel(self, include_additional_poles: bool = False):
        """
        Compute the full kernel matrix including additional poles

        Parameters:
        - include_additional_poles (bool): If True, additional poles are included in the kernel matrix

        Returns:
        - np.ndarray: Full kernel matrix including all poles
        """
        kernel_full = np.hstack((self.kernel, self.kernel_explicit))

        if include_additional_poles:
            kernel_full = np.hstack((kernel_full, self.kernel_additional))

        return kernel_full

    def _full_grid(self, include_additional_poles: bool = False):
        """
        Compute the full grid including additional poles

        Parameters:
        - include_additional_poles (bool): If True, additional poles are included in the grid

        Returns:
        - np.ndarray: Full grid including all poles
        """
        grid_full = np.concatenate(
            (self.fine_grid_complex, self.explicit_poles)
        )

        if include_additional_poles:
            grid_full = np.concatenate(
                (grid_full, self.additional_poles)
            )
        return grid_full

    def get_shared_attributes(self) -> dict:
        """
        Returns all attributes from the KernelMatrix base class.
        This is useful when initializing one of the inherited classes with
        an instance of another inherited class.

        Returns:
            dict: Dictionary containing all class attributes of KernelMatrix.
        """
        base_class_attributes = [
            "m",
            "n",
            "beta",
            "N_max",
            "delta_t",
            "h",
            "phi",
            "times",
            "fine_grid",
            "k_values",
            "kernel",
            "spec_dens",
            "freq_parametrization",
            "only_positive_particle",
            "fine_grid_complex",
            "explicit_poles",
            "explicit_residues",
            "kernel_explicit",
            "additional_poles",
            "additional_residues",
            "kernel_additional",
        ]

        base_class_attrs = {
            key: getattr(self, key, None) for key in base_class_attributes
        }

        return base_class_attrs

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_keys = [
            "m",
            "n",
            "beta",
            "N_max",
            "delta_t",
            "h",
            "phi",
            "freq_parametrization",
            "only_positive_particle",
        ]

        param_dict = {key: getattr(self, key) for key in param_keys}

        return param_dict

    def discrete_integral(
        self, kernel: np.ndarray = None, include_additional_poles: bool = False
    ) -> np.ndarray:
        """
        Computes the discrete approximation to the frequency integral at the times defined on the time grid

        Parameters:
        - kernel (np.ndarray, optional): Kernel matrix, where different rows correspond to different time steps, and different columns correspond to different frequencies
        - include_additional_poles (bool): If True, additional poles are included when computing the discrete integral
        Returns:
        - np.ndarray: Discrete approximation result to frequency integral at times on time grid
        """

        # Use provided or default kernel, extended by explicit poles
        if kernel is not None:
            kernel_eff = np.hstack((kernel, self.kernel_explicit))
        else:
            kernel_eff = np.hstack((self.kernel, self.kernel_explicit))

        #if additional poles are to be included
        if include_additional_poles:
            kernel_eff = np.hstack((kernel_eff, self.kernel_additional))


        if not isinstance(kernel_eff, np.ndarray):
            raise TypeError(
                f"'kernel' must be of type np.ndarray. Found {type(kernel_eff).__name__}"
            )

        # Sum over the frequency axis
        propag = np.sum(kernel_eff, axis=1).flatten()

        return propag

"""
Module to compute the kernel based on the AAA decomposition of the spectral density and Fermi-Dirac distribution.
"""
import numpy as np
from src.kernel_params.kernel_params import KernelParams
import src.utils.common_funcs as cf
from src.AAA.aaa_algorithm import aaa, cleanup


class AAARep:

    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        h: float,
        spec_dens: callable,
        freq_parametrization: str,
        tol: float = 1.e-13,
        mmax: int = 100,
        **kwargs
    ):
        # check if all parameters are valid
        KernelParams.validate_m_n(m, n)
        KernelParams.validate_beta(beta)
        KernelParams.validate_h(h)
        KernelParams.validate_freq_parametrization(freq_parametrization)

        for kwarg in kwargs:# ignore if an upper cutoff is specified as this is only relevant when computing continuous frequency integral as in DiscrKernel
            if kwarg in ["upper_cutoff", "phi", "N_max", "delta_t"]: #these keywords are not needed for the AAARep
                pass
            else:
                raise ValueError(f"Invalid keyword argument: {kwarg}")
    

        # Store parameters
        self.m, self.n = m, n
        self.beta = beta
    
        self.h = h
        self.spec_dens = spec_dens
        self.freq_parametrization = freq_parametrization
        self.tol = tol

        # initialize frequency grid: return fine grid (positive freqs), and Jacobian (for positive freqs)
        fine_grid, _, _ = cf.initialize_fine_grid(self.m, self.n, self.h, self.freq_parametrization)
   
        #full frequency grid including also negative frequencies:
        self.Z = np.concatenate((-fine_grid[::-1], fine_grid))

        #particle contribution (beta -> -beta) for hole distribution
        self.F_particle = cf.fermi_dirac(self.Z, beta = -self.beta) * self.spec_dens(self.Z)
        #hole contribution 
        self.F_hole = cf.fermi_dirac(self.Z, beta = self.beta) * self.spec_dens(self.Z)
       
        #perform AAA algorithm on spectral density multiplied with Fermi-Dirac distribution
        #particle contribution
        self.r_particle, self.errors_particle = aaa(Z = self.Z, F = self.F_particle, return_errors=True, tol = self.tol,  mmax = np.min([mmax, 2*(self.m + self.n) + 1]))# if default argument for maximal iterations is not sufficient, increase. Maximal allowed value is: mmax = 2*(self.m + self.n) + 1
        #hole contribution
        self.r_hole, self.errors_hole = aaa(Z = self.Z, F = self.F_hole, return_errors=True, tol = self.tol,  mmax = np.min([mmax, 2*(self.m + self.n) + 1]))# if default argument for maximal iterations is not sufficient, increase. Maximal allowed value is: mmax = 2*(self.m + self.n) + 1
        
        #determine poles and residues of the rational approximations
        self.poles_particle, self.residues_particle = self.r_particle.polres()
        self.poles_hole, self.residues_hole = self.r_hole.polres()

        #number of poles in the upper half plane
        self.nbr_poles_upper = np.sum(np.imag(self.poles_particle) > 0) + np.sum(np.imag(self.poles_hole) > 0)


    def remove_Froissart(self) -> None:
        """
        Remove potential Froissart doublets from the rational approximations.
        """
        self.r_particle = cleanup(self.r_particle, self.Z, self.F_particle)
        self.r_hole = cleanup(self.r_hole, self.Z, self.F_hole)

        #update poles and residues
        self.poles_particle, self.residues_particle = self.r_particle.polres()
        self.poles_hole, self.residues_hole = self.r_hole.polres()
        #number of poles in the upper half plane
        self.nbr_poles_upper = np.sum(np.imag(self.poles_particle) > 0) + np.sum(np.imag(self.poles_hole) > 0)


    def rational_approx(self, omega: float) -> tuple:
        """
        Compute the rational approximation of the spectral density multiplied with the Fermi-Dirac distribution at a given frequency omega.

        Parameters:
        - omega (float): Frequency argument for the propagator.

        Returns:
        - tuple: Rational approximations for the particle and hole contributions.
        """
        return (self.r_particle(omega), self.r_hole(omega))
    
    def polres(self) -> tuple:
        """
        Return poles and residues of the rational approximations for the particle and hole contributions.

        Parameters:
        - None

        Returns:
        - tuple: Poles and residues for the particle and hole contributions.
        """
    
        return (self.r_particle.polres(), self.r_hole.polres())
    
    def propagator_AAA(self, time: np.ndarray = None) -> tuple:
        """
        Compute the propagator for a given set of time steps.

        Parameters:
        - time (np.ndarray): Time argument for the propagator.

        Returns:
        - tuple: Propagator for the particle and hole contributions (concatenated to a single array).
        """
        kernel_particle, kernel_hole = self.build_kernel(time)

        #compute particle and hole propagators via residue theorem
        G_particle = np.sum(kernel_particle, axis=1).flatten()
        G_hole = np.sum(kernel_hole, axis=1).flatten()

        return np.concatenate((G_particle, G_hole))
    
    def upper_polres(self):
        """
        Determine the poles in the upper half plane and the correspondign residues

        Parameters:
        - None

        Returns:
        - tuple: poles_particle_upper, residues_particle_upper, poles_hole_upper, residues_hole_upper
        """
        
        #determine poles in the upper half plane and the corresponding residues
        #particles
        particle_mask = np.imag(self.poles_particle) > 0#mask for poles in the upper half plane
        poles_particle_upper = self.poles_particle[particle_mask]
        residues_particle_upper = self.residues_particle[particle_mask]

        #holes
        hole_mask = np.imag(self.poles_hole) > 0
        poles_hole_upper = self.poles_hole[hole_mask]
        residues_hole_upper = self.residues_hole[hole_mask]

        return (poles_particle_upper, residues_particle_upper, poles_hole_upper, residues_hole_upper)
    
    def get_params(self):
            """
            Returns a dict containing the parameters associated with an instance of the class and stored as attributes
            """

            param_keys = ["m", "n", "beta", "h", "freq_parametrization"]

            param_dict = {key: getattr(self, key) for key in param_keys}

            return param_dict
    
    def build_kernel(self, timegrid: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix based on the AAA decomposition and the time grid.

        Parameters:
        - time_grid (np.ndarray): Time grid for the kernel matrix.

        Returns:
        - np.ndarray: Kernel matrix.
        """
        #compute poles and residues in the upper half plane
        poles_particle_upper, residues_particle_upper, poles_hole_upper, residues_hole_upper = self.upper_polres()

        kernel_particle = 2.j * np.pi * residues_particle_upper  * np.exp(1.j * poles_particle_upper * timegrid[:,np.newaxis])
        kernel_hole = 2.j * np.pi * residues_hole_upper  * np.exp(1.j * poles_hole_upper * timegrid[:,np.newaxis])

        return kernel_particle, kernel_hole
        

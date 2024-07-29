"""
Module to compute the kernel based on the AAA decomposition of the spectral density and Fermi-Dirac distribution.
"""
import numpy as np
from src.kernel_params.kernel_params import KernelParams
import src.utils.common_funcs as cf
from src.AAA.aaa_algorithm import aaa, cleanup
from src.decomp_kernel.InterpolDecomp import InterpolDecomp


class AAARep:

    def __init__(
        self,
        fine_grid: np.ndarray,
        beta: float,
        spec_dens: callable,
        tol: float = 1.e-13,
        mmax: int = 100,
    ):
        """
        Parameters:
        - fine_grid (np.ndarray): Fine frequency grid for positive frequencies. Negative frequencies will be constructed by reflecting the positive frequencies at 0.
        - beta (float): Inverse temperature.
        - spec_dens (callable): Spectral density function.
        - tol (float): Tolerance for the AAA algorithm.
        - mmax (int): Maximal number of iterations for the AAA algorithm.
        """

        # check if all parameters are valid
        KernelParams.validate_beta(beta)

        # Store parameters as attributes
        self.beta = beta
        self.spec_dens = spec_dens
        self.tol = tol

        #check if fine grid contains only positive values
        if not np.all(fine_grid >= 0):
            raise ValueError("The fine grid must contain only positive values.")
        
        #sort fine grid in ascending order
        fine_grid = np.sort(fine_grid)

        #full frequency grid including also negative frequencies:
        self.Z = np.concatenate((-fine_grid[::-1], fine_grid))

        #particle contribution (beta -> -beta) for hole distribution
        self.F_particle = cf.fermi_dirac(self.Z, beta = -self.beta) * self.spec_dens(self.Z)
        #hole contribution 
        self.F_hole = cf.fermi_dirac(self.Z, beta = self.beta) * self.spec_dens(self.Z)
       
        #perform AAA algorithm on spectral density multiplied with Fermi-Dirac distribution
        #particle contribution
        self.r_particle, self.errors_particle = aaa(Z = self.Z, F = self.F_particle, return_errors=True, tol = self.tol,  mmax = np.min([mmax, len(self.Z)]))# if default argument for maximal iterations is not sufficient, increase. Maximal allowed value is: mmax = 2*(self.m + self.n) + 1
        #hole contribution
        self.r_hole, self.errors_hole = aaa(Z = self.Z, F = self.F_hole, return_errors=True, tol = self.tol,  mmax = np.min([mmax, len(self.Z)]))# if default argument for maximal iterations is not sufficient, increase. Maximal allowed value is: mmax = 2*(self.m + self.n) + 1
        
        #determine poles and residues of the rational approximations
        self.poles_particle, self.residues_particle = self.r_particle.polres()
        self.poles_hole, self.residues_hole = self.r_hole.polres()

        #compute poles and residues in the upper half plane
        self.poles_particle_upper, self.residues_particle_upper, self.poles_hole_lower, self.residues_hole_lower = self._eff_polres()

        #number of effective poles (in upper plane for particles and in lower plane for holes)
        self.nbr_poles_upper = len(self.poles_particle_upper) + len(self.poles_hole_lower)


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
    
    def _eff_polres(self):
        """
        Determine the effetive poles in the i) upper half plane for particles and ii) lower half plane for holes, and get the correspondign residues

        Parameters:
        - None

        Returns:
        - tuple: poles_particle_upper, residues_particle_upper, poles_hole_lower, residues_hole_lower
        """
        
        #determine poles in the upper half plane and the corresponding residues
        #particles
        particle_mask = np.imag(self.poles_particle) > 0#mask for poles in the upper half plane
        poles_particle_upper = self.poles_particle[particle_mask]
        residues_particle_upper = self.residues_particle[particle_mask]

        #holes
        hole_mask = np.imag(self.poles_hole) < 0
        poles_hole_lower = self.poles_hole[hole_mask]
        residues_hole_lower = self.residues_hole[hole_mask]

        return (poles_particle_upper, residues_particle_upper, poles_hole_lower, residues_hole_lower)
    
    def get_params(self):
            """
            Returns the temperature as dictionary (for compatibility with KernelMatrix class)
            """

            return {"beta": self.beta}
    
    def build_kernel(self, time_grid: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix based on the AAA decomposition and the time grid.

        Parameters:
        - time_grid (np.ndarray): Time grid for the kernel matrix.

        Returns:
        - np.ndarray: Kernel matrix.
        """

        kernel_particle = 2.j * np.pi * self.residues_particle_upper  * np.exp(1.j * self.poles_particle_upper * time_grid[:,np.newaxis])
        kernel_hole = - 2.j * np.pi * self.residues_hole_lower  * np.exp(-1.j * self.poles_hole_lower * time_grid[:,np.newaxis])

        return kernel_particle, kernel_hole
    

    def compress(self,
        time_grid : np.ndarray,
        eps: float = 1.e-15,
        compute_SVD = False) -> None:
        """
        Compress AAA kernel using ID and, if requested, SVD.

        Parameters:
        - time_grid (np.ndarray): Time grid for the kernel matrix.
        - eps (float): Error threshold for ID.
        - compute_SVD (bool): Flag to compute SVD.

        Returns:
        - None
        """
        #create kernel matrices for particles and holes
        kernel_particle, kernel_hole = self.build_kernel(time_grid)

        #create object of type InterpolDecomp for particles and holes, respectively.
        ID_particle = InterpolDecomp(kernel_particle, full_grid = self.poles_particle_upper, eps = eps, compute_SVD = compute_SVD)
        ID_hole = InterpolDecomp(kernel_hole, full_grid = self.poles_hole_lower, eps = eps, compute_SVD = compute_SVD)

        #store ID objects
        self.ID_particle = ID_particle
        self.ID_hole = ID_hole

        #store ID ranks, poles and residues as attributes
        self.ID_rank_particle = ID_particle.ID_rank
        self.ID_rank_hole = ID_hole.ID_rank

        #poles in upper plane as chosen by ID
        self.ID_poles_particle_upper = self.ID_particle._compute_coarse_grid()
        self.ID_poles_hole_lower = self.ID_hole._compute_coarse_grid()

        #residues in upper plane as chosen by ID
        self.ID_residues_particle_upper, self.ID_residues_hole_lower = self._residues_ID()
    
    def _residues_ID(self) -> tuple:
        """
        Return the residues of the compressed kernel matrix.

        Parameters:
        - None

        Returns:
        - tuple: eff_residues_particle, eff_residues_hole
        """
        #if ID objects are not yet computed, raise an error
        if not hasattr(self, 'ID_particle'):
            raise ValueError("The kernel matrix has not been compressed yet. Please call the 'compress' method first.")

        idx_particle = self.ID_particle.idx
        idx_hole = self.ID_hole.idx

        ID_rank_particle = self.ID_particle.ID_rank
        ID_rank_hole = self.ID_hole.ID_rank

        eff_residues_particle = self.residues_particle_upper[idx_particle[:ID_rank_particle]]
        eff_residues_hole = self.residues_hole_lower[idx_hole[:ID_rank_hole]]

        return (eff_residues_particle, eff_residues_hole)
        

    def propagator_AAA_compressed(self) -> np.ndarray:

        """
        Returns the propagator computed from the compressed kernel matrix.

        Parameters:
        - None

        Returns:
        - np.ndarray: Propagator for the particle and hole contributions (concatenated to a single array).
        """

        #if ID objects are not yet computed, raise an error
        if not hasattr(self, 'ID_particle'):
            raise ValueError("The kernel matrix has not been compressed yet. Please call the 'compress' method first.")
        
        #compute propagator from compressed kernel matrices
        G_particle = self.ID_particle.reconstruct_propagator_ID()
        G_hole = self.ID_hole.reconstruct_propagator_ID()

        return np.concatenate((G_particle, G_hole))
       


        

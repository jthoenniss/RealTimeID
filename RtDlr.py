import numpy as np
import RtKernel as ker
import scipy.linalg.interpolative as sli


class RtDlr:
    def __init__(
        self, m=None, n=None, beta=None, times=None, eps=None, h=None, phi=None
    ):
        """
        Parameters:
        - m (int): number of discretization intervals for omega > 1
        - n (int): number of discretization intervals for omega < 1
        - beta (float): inverse temperature
        - times (numpy.ndarray): Array containing the points on the time grid
        - eps (float): error for interpolvative decomposition (ID) and singular value decomposition (SVD)
        - h (float): Discretization parameter
        - phi (float): rotation angle in complex plane
        """
        if isinstance(
            m, ker.DiscrError
        ):  # Check if m is an object of the class DiscrError
            # Extract values from the DiscrError object
            members_DiscrError = ["m", "n", "beta", "times", "eps", "h", "phi"]
            # Copy variables from DiscrError object
            for member in members_DiscrError:
                setattr(self, member, getattr(m, member))
        else:
            # Use the explicitly provided values
            self.m = m
            self.n = n
            self.beta = beta
            self.times = times
            self.eps = eps
            self.h = h
            self.phi = (
                phi if phi is not None else np.pi / 4
            )  # Use provided phi or default value

        # __________________________________________________
        # Initialize RtKernel object with given parameters
        opts = {
            "m": self.m,
            "n": self.n,
            "beta": self.beta,
            "times": self.times,
            "eps": self.eps,
            "h": self.h,
            "phi": self.phi,
        }

        # Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank (see documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html" (access: 6. Dec. 2023))
        assert (
            self.eps < 1
        ), "'eps' needs to be smaller than 1 to be interpreted as an error and not as a rank (see scipy documentation for ID)"

        rt_kernel = ker.RtKernel(
            **opts
        )  # local object, not accessible outside __init__

        # __________________________________________________
        # Copy the state variables from the RtKernel object
        members_RtKernel = [
            "fine_grid",
            "num_singular_values_above_threshold",
            "singular_values",
            "ID_rank",
            "idx",
            "proj",
            "coarse_grid",
            "K",
        ]

        # Copy variables from RtKernel instance "rt_kernel"
        for member in members_RtKernel:
            setattr(self, member, getattr(rt_kernel, member))
        # ____________________________________________________

    def get_coarse_grid(self):
        """
        Return coarse frequency grid containing the frequencies chosen by the interpolative decomposition
        """
        return self.coarse_grid

    def spec_dens_fine(self, phi=None):
        """
        Evaluate the spectral density at the frequency points of the fine grid
        Parameters:
        - phi (float): rotation angle in complex plane

        Returns:
        - np.complex: spectral density evaluated at complex frequencys of fine grid (rotated into complex plane)
        """
        if phi is None:
            phi_cmplx = self.phi
        else:
            phi_cmplx = phi

        spec_dens_at_fine_grid = np.array(
            [ker.spec_dens(w_f * np.exp(1.0j * phi_cmplx)) for w_f in self.fine_grid]
        )
        return spec_dens_at_fine_grid

    def get_projection_matrix(self):
        """
        Compute the projection matrix needed to compute effective couplings
        """
        P = np.hstack([np.eye(self.ID_rank), self.proj])[
            :, np.argsort(self.idx)
        ]  # projection matrix

        return P

    def coupl_eff(self):
        """
        Compute effective couplings: multiply vector of spectral density at fine grid points with projection matrix P.
        """
        P = self.get_projection_matrix()
        coupl_eff = P @ self.spec_dens_fine()
        return coupl_eff

    def get_error(self):
        """
        Returns:
        - (DiscError): error object w.r.t. continous frequencey integration
        """
        return ker.DiscrError(
            self.m, self.n, self.N_max, self.delta_t, self.beta, self.upper_cutoff
        )

    def reconstr_interp_matrix(self):
        """
        Parameters:

        Returns:
        2D matrix with np.complex_: ID reconstructed matrix
        """

        # _____reconstruct interpolation matrix_________
        B = sli.reconstruct_skel_matrix(self.K, self.ID_rank, self.idx)
        # reconstructed interpolation matrix:
        K_reconstr = sli.reconstruct_matrix_from_id(B, self.idx, self.proj)

        return K_reconstr

    def reconstruct_propag(self, compute_error=False):
        """
        Reconstruct the propagator with ID approximation. Optionally also compute error to original propagator
        Parameters:
        compute_error (bool): flag that decides wheather the relative time-integrated error to original propagator is computed

        Returns:
        - np.array(np.complex_): propagator at all time points of fine grid
        - float: relative error between original and reconstructed Green's function [only if compute_error flag is set to 'True']
        """
        K_reconstr = self.reconstr_interp_matrix()  # ID-reconstructed kernel matrix
        Gamma = self.spec_dens_fine()  # spectral denstiy evaluated on fine grid points

        G_reconstr = (
            K_reconstr @ Gamma
        )  # this yields the propagators where the array elements correspond to the different time points

        yield G_reconstr  # return reconstructed Green's function

        if compute_error:  # evaluate error if flag is true
            G_orig = self.K @ Gamma

            error_rel = np.sum(abs(G_orig - G_reconstr)) / np.sum(
                abs(G_orig) + abs(G_reconstr)
            )  # in the relative error, the time steps cancels out and is thus not needed here.

            yield error_rel

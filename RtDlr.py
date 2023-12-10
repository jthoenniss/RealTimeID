import numpy as np
import RtKernel as ker
import scipy.linalg.interpolative as sli

class RtDlr:
    def __init__(
        self, N_max, delta_t, beta, upper_cutoff, m, n, eps=None, phi=np.pi / 4, h=None
    ):
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
        if (
            eps is None
        ):  # if no error is specified, use relative error vs frequency discretized result
            self.eps = ker.DiscrError(
                m, n, N_max, delta_t, beta, upper_cutoff
            ).error_time_integrated()
        else:
            self.eps = eps

        if h is None:
            self.h = (
                np.log(upper_cutoff) / m
            )  # choose discretization parameter h such that the highest frequency is the upper_cutoff
        else:
            self.h = h

        self.N_max = N_max
        self.delta_t = delta_t
        self.beta = beta
        self.upper_cutoff = upper_cutoff
        self.m = m
        self.n = n
        self.phi = phi

        # define dictionary with parameters needed to initialize RtKernel object
        opts = dict(
            N_max=self.N_max,
            delta_t=self.delta_t,
            beta=self.beta,
            upper_cutoff=self.upper_cutoff,
            m=self.m,
            n=self.n,
            eps=self.eps,
            h=self.h,
            phi=self.phi,
        )

        # Initialize RtKernel object with given parameters
        # Important: the variable "eps" needs to be smaller than 1 to be interpreted as an error and not as a rank (see documentation on "https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html" (access: 6. Dec. 2023))
        assert (
            eps < 1
        ), "'eps' needs to be smaller than 1 to be interpreted as an error and not as a rank (see scipy documentation for ID)"
        rt_kernel = ker.RtKernel(
            **opts
        )  # local object, not accessible outside __init__

        members = [
            "fine_grid",
            "times",
            "num_singular_values_above_threshold",
            "singular_values",
            "ID_rank",
            "idx",
            "proj",
            "coarse_grid",
            "times",
            "K",
        ]

        # Copy variables from RtKernel instance "rt_kernel"
        for member in members:
            setattr(self, member, getattr(rt_kernel, member))

    def get_coarse_grid(self):
        """
        Return coarse frequency grid containing the frequencies chosen by the interpolative decomposition
        """
        return self.coarse_grid

    def spec_dens_fine(self, phi = None):
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

        spec_dens_at_fine_grid = np.array([ker.spec_dens(w_f * np.exp(1.j * phi_cmplx)) for w_f in self.fine_grid])
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

        #_____reconstruct interpolation matrix_________
        B = sli.reconstruct_skel_matrix(self.K, self.ID_rank, self.idx)
        #reconstructed interpolation matrix:
        K_reconstr = sli.reconstruct_matrix_from_id(B, self.idx, self.proj)

        return K_reconstr
    

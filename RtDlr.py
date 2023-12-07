import numpy as np
import RtKernel as ker


class RtDlr:
    def __init__(
        self, N_max, delta_t, beta, cutoff, m, n, eps=None, phi=np.pi / 4, h=None
    ):
        """
        Parameters:
        - N_max (int): nbr. of time steps up to final time
        - delta_t (float): time discretization step
        - beta (float): inverse temperature
        - cutoff (float): maximal energy considered
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
                m, n, N_max, delta_t, beta, cutoff
            ).error_time_integrated()
        else:
            self.eps = eps

        if h is None:
            self.h = (
                np.log(cutoff) / m
            )  # choose discretization parameter h such that the highest frequency is the cutoff
        else:
            self.h = h

        self.N_max = N_max
        self.delta_t = delta_t
        self.beta = beta
        self.cutoff = cutoff
        self.m = m
        self.n = n
        self.phi = phi

        # define dictionary with parameters needed to initialize RtKernel object
        opts = dict(
            N_max=self.N_max,
            delta_t=self.delta_t,
            beta=self.beta,
            cutoff=self.cutoff,
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
            "P",
            "coarse_grid",
            "times",
        ]

        # Copy variables from RtKernel instance "rt_kernel"
        for member in members:
            setattr(self, member, getattr(rt_kernel, member))

    def get_coarse_grid(self):
        """
        Return coarse frequency grid containing the frequencies chosen by the interpolative decomposition
        """
        return self.coarse_grid

    def spec_dens_fine(self):
        """
        Evaluate the spectral density at the frequency points of the fine grid
        """
        spec_dens_at_fine_grid = [ker.spec_dens(w_f) for w_f in self.fine_grid]
        return spec_dens_at_fine_grid

    def coupl_eff(self):
        """
        Compute effective couplings: multiply vector of spectral density at fine grid points with projection matrix P.
        """
        coupl_eff = self.P @ self.spec_dens_fine()
        return coupl_eff

    def get_error(self):
        """
        Returns:
        - (DiscError): error object w.r.t. continous frequencey integration
        """
        return ker.DiscrError(
            self.m, self.n, self.N_max, self.delta_t, self.beta, self.cutoff
        )

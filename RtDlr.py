import numpy as np
import RtKernel as ker


class RtDlr:
    def __init__(self, t_max, delta_t, beta, cutoff, m, n, eps, phi=np.pi / 4):
        self.t_max = t_max
        self.delta_t = delta_t
        self.beta = beta
        self.cutoff = cutoff
        self.eps = eps
        self.m = m
        self.n = n
        # define dictionary with parameters needed to initialize RtKernel object
        opts = dict(
            t_max=t_max,
            delta_t=delta_t,
            beta=beta,
            cutoff=cutoff,
            m=m,
            n=n,
            eps=eps,
            phi=phi
        )

        # Initialize RtKernel object with given parameters
        rt_kernel = ker.RtKernel(**opts)  # local object, not accessible outside __init__

        members = [
            "h",
            "fine_grid",
            "times",
            "num_singular_values_above_threshold",
            "singular_values",
            "ID_rank",
            "idx",
            "P",
            "coarse_grid",
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
        return ker.DiscrError(self.m,self.n,self.t_max,self.delta_t,self.beta,self.cutoff)

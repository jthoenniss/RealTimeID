import numpy as np
import RtKernel as ker


class RtDlr:
    def __init__(self, N_max, delta_t, beta, cutoff, m, n, eps = None, phi=np.pi / 4):
        
        if eps is None:#if no error is specified, use absolute error vs frequency discretized result
            self.eps = ker.DiscrError(m,n,N_max,delta_t,beta,cutoff).abs_error_time_integrated()
        else:
            self.eps = eps

        self.N_max = N_max
        self.delta_t = delta_t
        self.beta = beta
        self.cutoff = cutoff
        self.m = m
        self.n = n
        self.phi = phi

        # define dictionary with parameters needed to initialize RtKernel object
        opts = dict(
            N_max = self.N_max,
            delta_t = self.delta_t,
            beta= self.beta,
            cutoff=self.cutoff,
            m=self.m,
            n=self.n,
            eps=self.eps,
            phi=self.phi
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
        return ker.DiscrError(self.m,self.n,self.N_max,self.delta_t,self.beta,self.cutoff)

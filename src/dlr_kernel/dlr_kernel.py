import numpy as np
import scipy.linalg.interpolative as sli
from src.utils import common_funcs as cf
from src.decomp_kernel.decomp_kernel import DecompKernel

class DlrKernel(DecompKernel):
    """
    Class providing an interface to class DecompKernel, extending its functionalities and providing access to effective quantities.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters are passed
        -- implicitly as attributes of an object, e.g. of type DiscrError: "dlr.RtDlr(discr_error)", or
        -- as keyword arguments: "dlr.RtDlr(m=10, n=20, beta=10., times=[1, 2], eps=0.3, h=0.2, spec_dens= lambda x : np.ones_like(np.atleast_1d(x)))".

        Parameters:
        - m, n, beta, N_max, delta_t, h, phi: Parameters for kernel matrix (see KernelMatrix).
        - eps (float): Error threshold for SVD and ID (see DecompKernel).
        - spec_dens (callable): Single-parameter function that ouputs the spectral density.
        """

        super().__init__(*args,**kwargs)


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
        coupl_eff = P @ self.spec_dens_array_cmplx
        return coupl_eff

    def reconstr_interp_matrix(self):
        """
        Parameters:

        Returns:
        2D matrix with np.complex_: ID reconstructed matrix
        """

        # __reconstruct kernel matrix__:
        B = sli.reconstruct_skel_matrix(self.kernel, self.ID_rank, self.idx)
        # reconstructed kernelmatrix:
        kernel_reconstr = sli.reconstruct_matrix_from_id(B, self.idx, self.proj)

        return kernel_reconstr

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
        Gamma = self.spec_dens_array_cmplx  # spectral density evaluated on fine grid points

        # yields the propagators where the array elements correspond to the different time points:
        G_reconstr = K_reconstr @ Gamma

        yield G_reconstr  # return reconstructed Green's function

        if compute_error:  # evaluate error if flag is true
            G_orig = self.kernel @ Gamma

            # in the relative error, the time steps cancels out and is thus not needed.
            error_rel = np.sum(abs(G_orig - G_reconstr)) / np.sum(
                abs(G_orig) + abs(G_reconstr)
            )

            yield error_rel

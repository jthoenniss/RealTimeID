import numpy as np
import scipy.linalg.interpolative as sli
from src.utils import common_funcs as cf



class InterpolDecomp:

    """
    Class for performing Singular Value Decomposition (SVD) and Interpolative Decomposition (ID)
    on a kernel matrix.
    """

    def __init__(
        self,
        kernel: np.ndarray,
        full_grid: np.ndarray,
        eps: float,
        compute_SVD = False
    ):
        """
        Initialize the RtKernel with kernel matrix parameters and an error threshold for SVD and ID.

        Parameters:
        - kernel (np.ndarray): Kernel matrix.
        - full_grid (np.ndarray): Fine grid, including also extra poles if necessary
        - eps (float): Error threshold for SVD and ID.
        - compute_SVD (bool): Flag to compute SVD.

        Returns:
        - None
        """
        self.kernel = kernel
        self.full_grid = full_grid
        self.eps = eps

        # Perform SVD and ID and set coarse grid
        self._initialize_ID()
        #if flag is True, initialize SVD
        self._initialize_SVD(compute_SVD=compute_SVD)
        

    def _initialize_ID(self):
        """
        Performs ID on the kernel matrix.
        """
        self.ID_rank, self.idx, self.proj = self.perform_ID()
        # compute coarse ID grid
        self.coarse_grid = self._compute_coarse_grid()

    def _initialize_SVD(self, compute_SVD: bool):
        """
        If flag 'compute_SVD' is true, initialize SVD
        """
        if compute_SVD:
            self.nbr_sv_above_eps, self.singular_values = self.perform_SVD()
        else:
            self.nbr_sv_above_eps = 0
            self.singular_values = np.array([])
    

    def perform_SVD(self, eps=None):
        """
        Perform SVD on the kernel matrix and count the number of singular values above the error threshold.

        Returns:
        Tuple[int, np.ndarray]: Count of singular values above threshold and array of singular values.
        """
        _eps = self.eps if eps is None else eps
        nbr_sv_above_eps, singular_values = cf.compute_singular_values(
            self.kernel, _eps
        )
        return nbr_sv_above_eps, singular_values

    def perform_ID(self, eps=None):
        """
        Perform interpolative decomposition (ID) on the kernel matrix using the error threshold.

        Returns:
        Tuple[int, np.ndarray, np.ndarray]: The rank of ID, indices, and projection matrix.
        """
        _eps = self.eps if eps is None else eps
        ID_rank, idx, proj = sli.interp_decomp(self.kernel, _eps, rand = False)
        
        return ID_rank, idx, proj
    
    def _compute_coarse_grid(self):
        """
        Compute the coarse grid consisting of frequencies selected by ID from the fine grid.

        Returns:
        np.ndarray: Coarse grid array.
        """
        print("full grid: ", self.full_grid.shape)
        # the coarse grid is a subset of the full frequency grid (with negative and positive frequencies)
        coarse_grid = np.array(self.full_grid[self.idx[:self.ID_rank]])
        print("coarse grid: ", coarse_grid.shape)

        return coarse_grid
    
    

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
        Compute effective couplings: sum over frequencies of the projection matrix
        """
        P = self.get_projection_matrix()
        coupl_eff = np.sum(P, axis = 1).flatten()
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
    
    def reconstruct_propagator_ID(self):
        """
        Reconstructs the propagator from the ID approximation.

        Args:
        - None

        Returns:
        - np.ndarray: Reconstructed propagator.

        """

        # Reconstruct the kernel matrix from the ID approximation
        K_reconstr = self.reconstr_interp_matrix()  
        # Reconstruct the propagator from the reconstructed kernel matrix
        G_reconstr = np.sum(K_reconstr, axis = 1).flatten()

        return G_reconstr
    
    def renormalize(self) -> None:
        """
        Renormalize all coupling by the effective spectral density. 
        Updates the kernel matrix.

        Parameters:
        - None

        Returns:
        - None
        """

        #update the kernel matrix with the selected coulmns
        self.kernel = sli.reconstruct_skel_matrix(self.kernel, self.ID_rank, self.idx)

        #update the full grid with the coarse grid
        self.full_grid = self.coarse_grid

        #get effective spectral density
        coupl_eff = self.coupl_eff()[np.newaxis,:]

        #multiply the columns of the reduced kernel matrix elementwise with the entries of the vector coupl_eff
        self.kernel *= coupl_eff

        #perform ID
        self._initialize_ID()


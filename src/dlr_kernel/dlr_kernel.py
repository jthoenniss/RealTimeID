import numpy as np
import scipy.linalg.interpolative as sli
from src.utils import common_funcs as cf
from src.dlr_kernel.parameter_validator import ParameterValidator
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.discr_error.discr_error import DiscrError

class DecompKernel(KernelMatrix): 
    REQUIRED_PARAMS = {"m","n","beta","N_max","delta_t","h","phi","eps"}
    """
    Class for performing Singular Value Decomposition (SVD) and Interpolative Decomposition (ID)
    on a kernel matrix, extending the functionalities of KernelMatrix.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ): 
        
        """
        Initialize the RtKernel with kernel matrix parameters and an error threshold for SVD and ID.

        Parameters:
        - m, n, beta, N_max, delta_t, h, phi: Parameters for kernel matrix (see KernelMatrix).
        - eps (float): Error threshold for SVD and ID.
        """

        if args:
            if isinstance(args[0], DiscrError):
                self._initialize_from_DiscrError(args[0])
            else:
                raise ValueError(f"No known method to initiliaze DecompKernel from object of type {type(args[0].__name__)}.")
                        
        elif kwargs:
            #Check that all required parameters (defined in RtDlr.REQUIRED_PARAMS) are present
            ParameterValidator.validate_required_params(kwargs, DecompKernel.REQUIRED_PARAMS)
            #read error and initialize.
            eps = kwargs.pop('eps', None)  # Extract 'eps' and remove it from kwargs
            ParameterValidator.validate_eps(eps)
            self.eps = eps
            #initialize base class
            super().__init__(**kwargs)
        else:
            raise ValueError("Arguments required for initialization not provided.")


        # Perform SVD and ID
        self._perform_decompositions()

        # compute coarse ID grid
        self.coarse_grid = self._compute_coarse_grid()

    # _______________End Initialization Routine___________________________

    def _initialize_from_DiscrError(self, D: DiscrError) -> None:
        """
        Initializes DecompKernel from an instance of DiscrError.
        Takes over all attributes from the shared base class KernelMatrix,
        as well as 'eps' held by the instance of DiscrError.
        Initilization of base class 'super().__init__' is not called in this case.
        """
        #Extract eps, validate, and initialize
        eps = D.eps
        ParameterValidator.validate_eps(eps)
        self.eps = eps

        params_KernelMatrix = D.get_shared_attributes()
        for key, value in params_KernelMatrix.items():
            setattr(self,key, value)
       
    def _perform_decompositions(self):
        """
        Performs SVD and ID on the kernel matrix.
        """
        self.nbr_sv_above_eps, self.singular_values = self.perform_svd()
        self.ID_rank, self.idx, self.proj = self.perform_ID()
        

    def perform_svd(self, eps = None):
        """
        Perform SVD on the kernel matrix and count the number of singular values above the error threshold.

        Returns:
        Tuple[int, np.ndarray]: Count of singular values above threshold and array of singular values.
        """
        _eps = self.eps if eps is None else eps
        nbr_sv_above_eps, singular_values = cf.svd_check_singular_values(self.kernel, _eps)
        return nbr_sv_above_eps, singular_values

    def perform_ID(self, eps = None):
        """
        Perform interpolative decomposition (ID) on the kernel matrix using the error threshold.

        Returns:
        Tuple[int, np.ndarray, np.ndarray]: The rank of ID, indices, and projection matrix.
        """
        _eps = self.eps if eps is None else eps
        ID_rank, idx, proj = sli.interp_decomp(self.kernel, _eps)
        return ID_rank, idx, proj

    def _compute_coarse_grid(self):
        """
        Compute the coarse grid consisting of frequencies selected by ID from the fine grid.

        Returns:
        np.ndarray: Coarse grid array.
        """
        coarse_grid = np.array(self.fine_grid[self.idx[: self.ID_rank]])

        return coarse_grid


class DlrKernel(DecompKernel):
    """
    Class providing an interface to class DecompKernel, extending its functionalities and providing access to effective quantities.
    """
    #Parameters required for the initialization of the class
    REQUIRED_PARAMS = ["m", "n", "beta", "N_max", "delta_t", "eps", "h", "phi"]

    def __init__(self, *args, **kwargs):
        """
        Parameters are passed 
        -- implicitly as attributes of an object, e.g. of type DiscrError: "dlr.RtDlr(discr_error)", or
        -- as keyword arguments: "dlr.RtDlr(m=10, n=20, beta=10., times=[1, 2], eps=0.3, h=0.2)".

        Parameters:
        - m, n, beta, N_max, delta_t, h, phi: Parameters for kernel matrix (see KernelMatrix).
        - eps (float): Error threshold for SVD and ID (see DecompKernel).
        """

        if args:
            self._initialize_from_args(args)
        elif kwargs:
            self._validate_and_initialize(kwargs)
        else:
            self._initialize_with_defaults()



    # Function definitions used in initilization:
    def _initialize_from_args(self, args):
        """
        Initializes the object from the first argument in 'args'.

        The method supports this type of initialization in two ways:
        1. With defaults: Initializes with default values if the first argument is None.
        2. From an object: Uses attributes of the first argument.
        """
        
        if args[0] is None:
            self._initialize_with_defaults()
        else:
            self._initialize_from_object(args[0])

    def _initialize_from_object(self, obj):
        """
        Initialize the class from an object 'obj' (e.g. of type DiscrError) from which we extract the attributes.
        """
        if isinstance(obj, DiscrError):
            super().__init__(obj)
        
        else:
            init_params = {
                param: getattr(obj, param)
                for param in DlrKernel.REQUIRED_PARAMS
                if hasattr(obj, param)
            }  # check for each required attribute in RtDlr.REQUIRED_PARAMS

            # check if all required attributes are there, if yes, initialize, otherwise error will be thrown
            self._validate_and_initialize(init_params)  

    def _validate_and_initialize(self, params):
        """
        Validate parameters (i.e. check if all required arguments are present) and initialize the class.
        """ 
        ParameterValidator.validate_required_params(params, DlrKernel.REQUIRED_PARAMS)  
        super().__init__(**params)  

    def _initialize_with_defaults(self):
        """
        Private method to initialize with default values when no arguments are provided or argument is None.
        In this case, the initializer of the parent class Rt_Kernel is not called.
        """
        # Default initialization of integer, float, and array attributes
        integer_attributes = [
            "m",
            "n",
            "N_max",
            "nbr_sv_above_eps",
            "ID_rank",
        ]
        float_attributes = ["beta","delta_t", "eps", "h", "phi", "singular_values"]
        array_attributes = ["times", "fine_grid", "kernel", "idx", "proj", "coarse_grid"]

        for member in integer_attributes:
            setattr(self, member, 0)
        for member in float_attributes:
            setattr(self, member, 0.0)
        for member in array_attributes:
            setattr(self, member, np.array([]))

        # ____________________________________________________

    def get_coarse_grid(self):
        """
        Return coarse frequency grid containing the frequencies chosen by the interpolative decomposition
        """
        return self.coarse_grid

    def spec_dens_fine(self):
        """
        Evaluate the spectral density at the frequency points of the fine grid
        
        Returns:
        - numpy.ndarray: Spectral density evaluated at complex frequencies of the fine grid (rotated into the complex plane).
        """
        rotated_frequencies = self.fine_grid * np.exp(1.0j * self.phi)
        spec_dens_at_fine_grid = cf.spec_dens_array(rotated_frequencies)

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
        Gamma = self.spec_dens_fine()  # spectral density evaluated on fine grid points
        
        # yields the propagators where the array elements correspond to the different time points:
        G_reconstr = K_reconstr @ Gamma

        yield G_reconstr  # return reconstructed Green's function

        if compute_error:  # evaluate error if flag is true
            G_orig = self.kernel @ Gamma

            # in the relative error, the time steps cancels out and is thus not needed.
            error_rel = np.sum(abs(G_orig - G_reconstr)) / np.sum(abs(G_orig) + abs(G_reconstr))
            
            yield error_rel

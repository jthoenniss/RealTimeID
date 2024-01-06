import numpy as np
import scipy.linalg.interpolative as sli
from src.utils import common_funcs as cf
from src.kernel_params.parameter_validator import ParameterValidator
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.discr_error.discr_error import DiscrError


class DecompKernel(KernelMatrix):
    REQUIRED_PARAMS = {"m", "n", "beta", "N_max", "delta_t", "h", "phi", "eps", "spec_dens"}
    """
    Class for performing Singular Value Decomposition (SVD) and Interpolative Decomposition (ID)
    on a kernel matrix, extending the functionalities of KernelMatrix.
    """

    def __init__(
        self,
        *args,
        compute_SVD = False,
        **kwargs
    ):
        """
        Initialize the RtKernel with kernel matrix parameters and an error threshold for SVD and ID.

        Parameters:
        - m, n, beta, N_max, delta_t, h, phi: Parameters for kernel matrix (see KernelMatrix).
        - eps (float): Error threshold for SVD and ID.
        - spec_dens (callable): Single-parameter function that ouputs the spectral density.
        - compute_SVD (bool): Flag that determines whether the SVD or the kernel should be evaluated
        """

        if args:
            if args[0] is None:
                self._initialize_with_defaults()
            elif isinstance(args[0], DiscrError):
                self._initialize_from_DiscrError(args[0], compute_SVD)
            else:
                raise ValueError(
                    f"No known method to initialize DecompKernel from object of type {type(args[0]).__name__}."
                )

        elif kwargs:
            self._initialize_from_kwargs(kwargs, compute_SVD)
        
        else:
            raise ValueError("Arguments required for initialization not provided.")


    # _______________End Initialization Routine___________________________
    def _initialize_with_defaults(self) -> None:
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
        float_attributes = ["beta", "delta_t", "eps", "h", "phi", "singular_values"]
        array_attributes = [
            "times",
            "fine_grid",
            "kernel",
            "idx",
            "proj",
            "coarse_grid",
            "spec_dens_array_cmplx"
        ]

        for member in integer_attributes:
            setattr(self, member, 0)
        for member in float_attributes:
            setattr(self, member, 0.0)
        for member in array_attributes:
            setattr(self, member, np.array([]))
        
        self.spec_dens = None #this attribute is either None or a callable function that outputs the spectral density


    def _initialize_from_DiscrError(self, D: DiscrError, compute_SVD: bool) -> None:
        """
        Initializes DecompKernel from an instance of DiscrError.
        Takes over all attributes from the shared base class KernelMatrix,
        as well as 'eps' held by the instance of DiscrError.
        Initilization of base class 'super().__init__' is not called in this case.
        """
        # Extract eps, validate, and initialize
        eps = D.eps
        ParameterValidator.validate_eps(eps)
        self.eps = eps

        params_KernelMatrix = D.get_shared_attributes()

        for key, value in params_KernelMatrix.items():
            setattr(self, key, value)

        # Perform ID and set coarse grid
        self._initialize_ID()
        #if flag is True, initialize SVD
        self._initialize_SVD(compute_SVD=compute_SVD)

    def _initialize_from_kwargs(self,kwargs, compute_SVD: bool) -> None:
        # Check that all required parameters (defined in RtDlr.REQUIRED_PARAMS) are present
        ParameterValidator.validate_required_params(
            kwargs, DecompKernel.REQUIRED_PARAMS
        )
        # read error and initialize.
        eps = kwargs.pop("eps", None)  # Extract 'eps' and remove it from kwargs
        ParameterValidator.validate_eps(eps)
        self.eps = eps
        # initialize base class
        super().__init__(**kwargs)
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
    
    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_dict = super().get_params()

        param_dict["eps"] = getattr(self, "eps")#add parameters for eps which does not exist in base class KernelMatrix.

        return param_dict


import numpy as np
import scipy.linalg.interpolative as sli
from src.utils import common_funcs as cf
from src.kernel_params.kernel_params import KernelParams
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.discr_error.discr_error import DiscrError


class DecompKernel(KernelMatrix):
    REQUIRED_PARAMS = {
        "m",
        "n",
        "beta",
        "N_max",
        "delta_t",
        "h",
        "phi",
        "eps",
        "spec_dens",
        "freq_parametrization",
    }
    """
    Class for performing Singular Value Decomposition (SVD) and Interpolative Decomposition (ID)
    on a kernel matrix, extending the functionalities of KernelMatrix.
    """

    def __init__(
        self,
        *args,
        compute_SVD: bool = False,
        only_positive_particle: bool = False,
        explicit_poles: np.ndarray = np.empty(0),
        explicit_residues: np.ndarray = np.empty(0),
        additional_poles: np.ndarray = np.empty(0),
        additional_residues: np.ndarray = np.empty(0),
        include_additional_poles: bool = False,
        **kwargs,
    ):
        """
        Initialize the RtKernel with kernel matrix parameters and an error threshold for SVD and ID.

        Parameters:
        - m, n, beta, N_max, delta_t, h, phi: Parameters for kernel matrix (see KernelMatrix).
        - eps (float): Error threshold for SVD and ID.
        - spec_dens (callable): Single-parameter function that ouputs the spectral density.
        - compute_SVD (bool): Flag that determines whether the SVD or the kernel should be evaluated
        - only_positive_particle (bool): If True, only the positive particle is considered in the spectral density.
        - explicit_poles (np.ndarray): Array of explicit poles for the kernel matrix.
        - explicit_residues (np.ndarray): Array of explicit residues for the kernel matrix.
        - additional_poles (np.ndarray): Array of additional poles for the kernel matrix.
        - additional_residues (np.ndarray): Array of additional residues for the kernel matrix.
        - include_additional_poles (bool): If True, include additional poles in ID and SVD
        """

        if args:
            if args[0] is None:
                self._initialize_with_defaults()
            elif isinstance(args[0], DiscrError):
                self._initialize_from_DiscrError(args[0], compute_SVD, include_additional_poles)
            else:
                raise ValueError(
                    f"No known method to initialize DecompKernel from object of type {type(args[0]).__name__}."
                )

        elif kwargs:
            self._initialize_from_kwargs(
                kwargs, compute_SVD, only_positive_particle, explicit_poles, explicit_residues, additional_poles, additional_residues, include_additional_poles
            )

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
            "only_positive_particle",
            "include_additional_poles"
        ]
        float_attributes = ["beta", "delta_t", "eps", "h", "phi", "singular_values"]
        array_attributes = [
            "times",
            "fine_grid",
            "kernel",
            "idx",
            "proj",
            "coarse_grid",
            "fine_grid_complex",
            "explicit_poles",
            "explicit_residues",
            "additional_poles",
            "additional_residues",
            "kernel_additional",
        ]

        for member in integer_attributes:
            setattr(self, member, 0)
        for member in float_attributes:
            setattr(self, member, 0.0)
        for member in array_attributes:
            setattr(self, member, np.array([]))

        self.spec_dens = None  # this attribute is either None or a callable function that outputs the spectral density
        self.freq_parametrization = None  # this attribute is either None or a string that specifies the frequency parametrization

    def _initialize_from_DiscrError(self, D: DiscrError, compute_SVD: bool, include_additional_poles: bool) -> None:
        """
        Initializes DecompKernel from an instance of DiscrError.
        Takes over all attributes from the shared base class KernelMatrix,
        as well as 'eps' held by the instance of DiscrError.
        Initilization of base class 'super().__init__' is not called in this case.
        """
        # Extract eps, validate, and initialize
        eps = D.eps
        KernelParams.validate_eps(eps)
        self.eps = eps

        params_KernelMatrix = D.get_shared_attributes()

        for key, value in params_KernelMatrix.items():
            setattr(self, key, value)

        self.include_additional_poles = include_additional_poles
        # Perform ID and set coarse grid
        self._initialize_ID()
        # if flag is True, initialize SVD
        self._initialize_SVD(compute_SVD=compute_SVD)

    def _initialize_from_kwargs(
        self,
        kwargs,
        compute_SVD: bool,
        only_positive_particle: bool,
        explicit_poles: np.ndarray,
        explicit_residues: np.ndarray,
        additional_poles: np.ndarray,
        additional_residues: np.ndarray,
        include_additional_poles: bool
    ) -> None:
        # Check that all required parameters (defined in RtDlr.REQUIRED_PARAMS) are present
        KernelParams.validate_required_params(kwargs, DecompKernel.REQUIRED_PARAMS)
        # read error and initialize.
        eps = kwargs.pop("eps", None)  # Extract 'eps' and remove it from kwargs
        KernelParams.validate_eps(eps)
        self.eps = eps
        # initialize base class
        super().__init__(
            **kwargs,
            only_positive_particle=only_positive_particle,
            explicit_poles=explicit_poles,
            explicit_residues=explicit_residues,
            additional_poles=additional_poles,
            additional_residues=additional_residues,
        )

        self.include_additional_poles = include_additional_poles

        # Perform SVD and ID and set coarse grid
        self._initialize_ID()
        # if flag is True, initialize SVD
        self._initialize_SVD(compute_SVD=compute_SVD)

    def _initialize_ID(self, include_additional_poles: bool = None):
        """
        Performs ID on the kernel matrix.

        Parameters:
        - include_additional_poles(bool): If not None, additional poles are included according to the value of this parameter. Otherwise, the value of self.include_additional_poles is used.
        """
        self.ID_rank, self.idx, self.proj = self.perform_ID(include_additional_poles=include_additional_poles)
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

    def perform_SVD(self, eps=None) -> tuple:
        """
        Perform SVD on the kernel matrix and count the number of singular values above the error threshold.

        Parameters:
        - eps (float): SVD error
        - include_additional_poles(bool): If True, ID is performed on "full" kernel matrix, including the additional poles

        Returns:
        Tuple[int, np.ndarray]: Count of singular values above threshold and array of singular values.
        """
        _eps = self.eps if eps is None else eps

        kernel = self._full_kernel(include_additional_poles=self.include_additional_poles)

        nbr_sv_above_eps, singular_values = cf.compute_singular_values(kernel, _eps)
        return nbr_sv_above_eps, singular_values

    def perform_ID(self, eps=None, include_additional_poles: bool = None) -> tuple:
        """
        Perform interpolative decomposition (ID) on the kernel matrix using the error threshold.

        Parameters:
        - eps (float): ID error
        - include_additional_poles(bool): If not None, additional poles are included according to the value of this parameter. Otherwise, the value of self.include_additional_poles is used.
        Returns:
        Tuple[int, np.ndarray, np.ndarray]: The rank of ID, indices, and projection matrix.
        """
        _eps = self.eps if eps is None else eps

        _include_additional_poles = self.include_additional_poles if include_additional_poles is None else include_additional_poles

        kernel = self._full_kernel(include_additional_poles=_include_additional_poles)

        ID_rank, idx, proj = sli.interp_decomp(kernel, _eps, rand=False)

        return ID_rank, idx, proj

    def _compute_coarse_grid(self):
        """
        Compute the coarse grid consisting of frequencies selected by ID from the fine grid.

        Returns:
        np.ndarray: Coarse grid array.
        """

        fine_grid_complex_full = self._full_grid(include_additional_poles=self.include_additional_poles)

        print("full grid: ", fine_grid_complex_full[0], fine_grid_complex_full[-1])

        coarse_grid = fine_grid_complex_full[self.idx[: self.ID_rank]]

        return coarse_grid

    def get_params(self):
        """
        Returns a dict containing the parameters associated with an instance of the class and stored as attributes
        """

        param_dict = super().get_params()

        param_dict["eps"] = getattr(
            self, "eps"
        )  # add parameters for eps which does not exist in base class KernelMatrix.

        return param_dict

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
        coupl_eff = np.sum(P, axis = 1).flatten()
        return coupl_eff

    def reconstr_interp_matrix(self):
        """
        Parameters:

        Returns:
        2D matrix with np.complex_: ID reconstructed matrix
        """

        kernel = self._full_kernel(include_additional_poles=self.include_additional_poles)

        B = sli.reconstruct_skel_matrix(kernel, self.ID_rank, self.idx)
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
        kernel_full = self._full_kernel(include_additional_poles=self.include_additional_poles)
        self.kernel = sli.reconstruct_skel_matrix(kernel_full, self.ID_rank, self.idx)

        #update the full grid with the coarse grid
        self.fine_grid_complex = self.coarse_grid

        #get effective spectral density
        coupl_eff = self.coupl_eff()[np.newaxis,:]
   
        #multiply the columns of the reduced kernel matrix elementwise with the entries of the vector coupl_eff
        self.kernel *= coupl_eff
        
        #perform ID
        self._initialize_ID(include_additional_poles=False)


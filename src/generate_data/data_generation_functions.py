# Standard libraries
import numpy as np

# Import Custom Modules
from src.utils.module_utils.all_custom_modules import (
    DiscrError,
    DecompKernel,
    cf,
    Hdf5Kernel,
)  # Consolidated custom modules import
from src.kernel_params.kernel_params import KernelParams


def compute_grid_and_store(
    h_vals, N_maxs, betas, params: KernelParams, h5_kernel: Hdf5Kernel, optimize: bool = False, rel_error_diff: float = None
) -> None:
    """
    Compute discretization error and store results in an HDF5 file.

    This function computes the discretization error and the corresponding DlrKernel object for each point on a data grid
    and stores the resulting data in an HDF5 file associated with 'h5_kernel'.

    Parameters:
        h_vals (array type): Array of discretization parameter values to be evaluated.
        N_maxs (array type): Array of total number of time steps values to be evaluated.
        betas (array type): Array of inverse temperature values to be evaluated.
        params (KernelParams): An instance of KernelParams that holds the parameter set.
        h5_kernel (Hdf5Kernel): An instance of Hdf5Kernel associated with the HDF5 file for storing the results.
        optimize (bool, optional): If True, the values for m and n are optimized to reduce the frequency interval with addional error at most rel_error_diff of discretization error.
        rel_error_diff (float, optional): If optimize is True, this is the relative error difference that is allowed between the optimized and unoptimized values for m and n. If not set, default value defined in function 'optimize'.
    Returns:
        None

    """
    for b, beta in enumerate(betas):
        params.update_parameters({"beta": beta})

        for tau, N_max in enumerate(N_maxs):
            params.update_parameters({"N_max": N_max})

            # set time grid
            times = cf.set_time_grid(
                N_max=params.get_param("N_max"), delta_t=params.get_param("delta_t")
            )
            # compute continous-frequency integral
            cont_integral = cf.cont_integral(
                t=times,
                beta=params.get_param("beta"),
                upper_cutoff=params.get_param("upper_cutoff"),
                spec_dens=params.get_param("spec_dens"),
            )

            for h, h_val in enumerate(h_vals):

                params.update_parameters(
                    {"h": h_val}
                )  # this automatically updates "m" and "n".

                # Create DiscrError object which holds the error w.r.t. to the continous results, and all associated parameters.
                discr_error = DiscrError(
                    **params.params, cont_integral_init=cont_integral
                )

                if optimize:
                    discr_error.optimize(rel_error_diff=rel_error_diff)  # optimize values for m and n

                cont_integral = discr_error.cont_integral_init
                # create DlrKernel object based on DiscrError object
                decomp_kernel = DecompKernel(discr_error)

                # compute reconstruction error (between reconstructed propagator and continuous-frequency propagator)
                propagator_reconstr = decomp_kernel.reconstruct_propagator()
                eps = discr_error.error_time_integrated(time_series_approx=propagator_reconstr)
            
                print(f"Discretization error: {discr_error.eps:.2e}, reconstruction error: {eps:.2e}.")

                # store to hdf5 file
                h5_kernel.append_kernel_element(decomp_kernel, (h, tau, b))
            

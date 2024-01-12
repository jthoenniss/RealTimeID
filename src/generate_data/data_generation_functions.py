# Standard libraries
import numpy as np

# Import Custom Modules
from src.utils.module_utils.all_custom_modules import (
    DiscrError,
    DlrKernel,
    cf,
    Hdf5Kernel,
)  # Consolidated custom modules import
from src.kernel_params.kernel_params import KernelParams


def compute_grid_and_store(
    h_vals, N_maxs, betas, params: KernelParams, h5_kernel: Hdf5Kernel, optimize: bool = False
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
        optimize (bool): If True, the values for m and n are optimized to reduce frequenc interval with addional error at most 1% of discretization error.

    Returns:
        None

    """
    for b, beta in enumerate(betas):
        for tau, N_max in enumerate(N_maxs):
            # update parameters
            params.update_parameters({"N_max": N_max})
            params.update_parameters({"beta": beta})

            # set time grid
            times = cf.set_time_grid(
                N_max=params.get("N_max"), delta_t=params.get("delta_t")
            )
            # compute continous-frequency integral
            cont_integral = cf.cont_integral(
                t=times,
                beta=params.get("beta"),
                upper_cutoff=params.get("upper_cutoff"),
                spec_dens=params.get("spec_dens"),
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
                    discr_error.optimize()  # optimize values for m and n

                # Exit the loop it error is below machine precision, otherwise, append to hdf5 file
                if discr_error.eps < 1.0e-16:
                    break
                else:
                    # create DlrKernel object based on DiscrError object
                    kernel = DlrKernel(discr_error)
                    # store to hdf5 file
                    h5_kernel.append_kernel_element(kernel, (h, tau, b))

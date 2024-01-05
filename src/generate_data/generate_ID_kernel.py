# Standard libraries
import numpy as np


# Import Custom Modules
from src.utils.module_utils.all_custom_modules import (
    DiscrError,
    DlrKernel,
    cf,
    Hdf5Kernel
)  # Consolidated custom modules import
from src.kernel_params.kernel_params import KernelParams


# _______________Compute data for all points on parameter grid___________________


def compute_grid_and_store(
    h_vals, N_maxs, betas, params: KernelParams, h5_kernel: Hdf5Kernel
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
            cont_integral = np.array(
                [
                    cf.cont_integral(
                        t,
                        beta=params.get("beta"),
                        upper_cutoff=params.get("upper_cutoff"),
                    )
                    for t in times
                ]
            )

            for h, h_val in enumerate(h_vals):
                params.update_parameters({"h": h_val})

                # Create DiscrError object which holds the error w.r.t. to the continous results, and all associated parameters.
                discr_error = DiscrError(
                    **params.params, cont_integral_init=cont_integral
                )
                # discr_error.optimize()  # optimize values for m and n

                # Exit the loop it error is below machine precision, otherwise, append to hdf5 file
                if discr_error.eps < 1.0e-14:
                    break
                else:
                    # create DlrKernel object based on DiscrError object
                    kernel = DlrKernel(discr_error)
                    # store to hdf5 file
                    h5_kernel.append_kernel_element(kernel, (h, tau, b))


if __name__ == "__main__":

    # _______________Set Parameter Grid (choose values to explore)____________________
    # Array specifying all values for the discreitzation parameter, h, that should be evaluated
    h_vals = np.logspace(-0.2, -2, 15)
    # Array specifying all values for the total number of time steps, N_max, that should be evaluated
    N_maxs = list(map(int, np.logspace(1, 5, 10)))
    # Array specifying all values for inverse temperature, beta, that should be evaluated
    betas = [0, 20, 100, 10000]

    # Define filename of hdf5 file holding the data
    filename = f"data/filename.h5"

    # Create instance of Hdf5Kernel to be associated with the file
    h5_kernel = Hdf5Kernel(filename=filename)

    # create hdf5 file to write to
    param_grid_dims = (len(h_vals), len(N_maxs), len(betas))
    h5_kernel.create_file(kernel_dims=param_grid_dims)

    # Create instance of KernelParams to hold the parameter set (initialize with default values)
    params = KernelParams()
    # compute data and write to file
    compute_grid_and_store(
        h_vals=h_vals, N_maxs=N_maxs, betas=betas, params=params, h5_kernel=h5_kernel
    )

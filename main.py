# Standard libraries
import numpy as np
# Import Custom Modules
from src.utils.module_utils.all_custom_modules import Hdf5Kernel
from src.kernel_params.kernel_params import KernelParams
from src.generate_data.data_generation_functions import compute_grid_and_store
from src.spec_dens.spec_dens import spec_dens_gapless, spec_dens_gapped_sym
import time


"""
A script to compute the interpolative decomposition with all corresponding quantities on a parameter grid
and store the data in a hdf5 file.

Execute from the terminal with the command "python3 -m src.generate_data.main".

To evaluate the hdf5 file, see jupyter notebook "/notebooks/analyze_ID_kernel.ipynb".
"""


if __name__ == "__main__":

    time_init = time.time()

    # _______________Set Parameter Grid (choose values to explore)____________________
    # Array specifying all values for the discreitzation parameter, h, that should be evaluated
    h_vals = np.logspace(-2, -.2, 15)
    # Array specifying all values for the total number of time steps, N_max, that should be evaluated
    N_maxs = list(map(int, np.logspace(1, 2, 10)))
    # Array specifying all values for inverse temperature, beta, that should be evaluated
    betas = [0, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4, 1.e5]

    # Define filename of hdf5 file holding the data
    filename = f"data/delta_t=0.1_gapless.h5"

    # Create instance of Hdf5Kernel to be associated with the file
    h5_kernel = Hdf5Kernel(filename=filename)

    # create hdf5 file to write to
    param_grid_dims = (len(h_vals), len(N_maxs), len(betas))
    h5_kernel.create_file(kernel_dims=param_grid_dims)

    # Create instance of KernelParams to hold the parameter set (initialize with default values unless keyword arguments are specified)
    params = KernelParams()

    print(f"Starting computation of data on parameter grid with dimensions {param_grid_dims}.")
    # compute data and write to file
    compute_grid_and_store(
        h_vals=h_vals, N_maxs=N_maxs, betas=betas, params=params, h5_kernel=h5_kernel, optimize=True, rel_error_diff=0.01
    )

    run_time = time.time() - time_init
    #convert runtime from seconds to hours, minutes, seconds
    hours = round(run_time // 3600)
    minutes = round((run_time % 3600) // 60)
    seconds = round(run_time % 60)

    print(f"Finished computation of data on parameter grid.\n Total runtime: {hours}h {minutes}m {seconds}s.")


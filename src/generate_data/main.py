# Standard libraries
import numpy as np
# Import Custom Modules
from src.utils.module_utils.all_custom_modules import Hdf5Kernel
from src.kernel_params.kernel_params import KernelParams
from src.generate_data.data_generation_functions import compute_grid_and_store

"""
A script to compute the interpolative decomposition with all corresponding quantities on a parameter grid
and store the data in a hdf5 file.

Execute from the terminal with the command "python3 -m src.generate_data.main".

To evaluate the hdf5 file, see jupyter notebook "/notebooks/analyze_ID_kernel.ipynb".
"""


if __name__ == "__main__":

    # _______________Set Parameter Grid (choose values to explore)____________________
    # Array specifying all values for the discreitzation parameter, h, that should be evaluated
    h_vals = np.logspace(-2, -.2, 15)
    # Array specifying all values for the total number of time steps, N_max, that should be evaluated
    N_maxs = list(map(int, np.logspace(1, 5, 10)))
    # Array specifying all values for inverse temperature, beta, that should be evaluated
    betas = [0, 20, 100, 10000]

    # Define filename of hdf5 file holding the data
    filename = f"data/delta_t=0.1_large.h5"

    # Create instance of Hdf5Kernel to be associated with the file
    h5_kernel = Hdf5Kernel(filename=filename)

    # create hdf5 file to write to
    param_grid_dims = (len(h_vals), len(N_maxs), len(betas))
    h5_kernel.create_file(kernel_dims=param_grid_dims)

    # Create instance of KernelParams to hold the parameter set (initialize with default values unless keyword arguments are specified)
    params = KernelParams()
    # compute data and write to file
    compute_grid_and_store(
        h_vals=h_vals, N_maxs=N_maxs, betas=betas, params=params, h5_kernel=h5_kernel, optimize=True, rel_error_diff=1.e-16
    )


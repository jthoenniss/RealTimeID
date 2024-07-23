# Standard libraries
import numpy as np

# Import Custom Modules
from src.utils.module_utils.all_custom_modules import Hdf5Kernel
from src.kernel_params.kernel_params import KernelParams
from src.generate_data.data_generation_functions import (
    compute_ID_grid_and_store,
    compute_AAA_grid_and_store,
)
from src.spec_dens.spec_dens import (
    spec_dens_gapless,
    spec_dens_gapped_sym,
    spec_dens_exp,
    spec_dens_semi_circle,
    SpecDensGapless
)
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
    N_maxs = list(map(int, np.logspace(1, 3.5, 15)))
    h_vals = np.logspace(-1.8, -0.05, 25)  # order from small to large
    # Array specifying all values for inverse temperature, beta, that should be evaluated
    betas = [0, 1.e4]  # , 1.e3, 1.e4, 1.e5]

    # Define filename of hdf5 file holding the data
    filename_ID = f"data/test_ID_Lambda100000_opt.h5"

    # Create instance of Hdf5Kernel to be associated with the file
    ID_h5_kernel = Hdf5Kernel(filename=filename_ID)

    # create hdf5 files to write to
    param_grid_dims_ID = (len(h_vals), len(N_maxs), len(betas))

    ID_h5_kernel.create_file(kernel_dims=param_grid_dims_ID)

    #spec_dens = lambda x: spec_dens_gapless(x, cutoff_lower=-20, cutoff_upper=-5, sharpness=2) + spec_dens_gapless(x, cutoff_lower=5, cutoff_upper=20, sharpness=2)
    spec_dens = SpecDensGapless(Lambda = 100000)
    #spec_dens = lambda x: abs(x) * np.exp(abs(x)/10)
    #spec_dens = lambda x: abs(x) * spec_dens_gapless(x, cutoff_lower=-20, cutoff_upper=20, sharpness=1)

    print(
        f"Starting computation of ID-data on parameter grid with dimensions {param_grid_dims_ID}."
    )

    # Create instance of KernelParams to hold the parameter set (initialize with default values unless keyword arguments are specified)
    params_ID = KernelParams(spec_dens=spec_dens, freq_parametrization="simple_exp")
    # compute data and write to file for ID
    compute_ID_grid_and_store(
        h_vals=h_vals,
        N_maxs=N_maxs,
        betas=betas,
        params=params_ID,
        h5_kernel=ID_h5_kernel,
        optimize=True,
        rel_error_diff=0.1,
        only_positive_particle=True,
    )

    run_time = time.time() - time_init
    # convert runtime from seconds to hours, minutes, seconds
    hours = round(run_time // 3600)
    minutes = round((run_time % 3600) // 60)
    seconds = round(run_time % 60)

    print(
        f"Finished computation of data on parameter grid.\n Total runtime: {hours}h {minutes}m {seconds}s."
    )

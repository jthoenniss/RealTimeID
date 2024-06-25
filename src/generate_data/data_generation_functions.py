# Standard libraries
import numpy as np

# Import Custom Modules
from src.utils.module_utils.all_custom_modules import (
    DiscrError,
    DecompKernel,
    cf,
    Hdf5Kernel,
    AAAKernel,
)  # Consolidated custom modules import
from src.kernel_params.kernel_params import KernelParams


def compute_ID_grid_and_store(
    h_vals,
    N_maxs,
    betas,
    params: KernelParams,
    h5_kernel: Hdf5Kernel,
    optimize: bool = False,
    rel_error_diff: float = None,
    only_positive_particle: bool = False,
) -> None:
    """
    Compute discretization error for ID and store results in an HDF5 file.

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
        only_positive_particle (bool, optional): If True, only the positive frequency part of the particle component is considered.
    Returns:
        None

    """


    for b, beta in enumerate(betas):
        params.update_parameters({"beta": beta})

        # set time grid for maximal time needed
        times = cf.set_time_grid(
            N_max=N_maxs[-1], delta_t=params.get_param("delta_t")
        )
        # compute continuous-frequency integral such that they are not recomputed for every value of h below

        #particle component
        cont_integral_particle = cf.cont_integral(
            t=times,
            beta=params.get_param("beta"),
            upper_cutoff=params.get_param("upper_cutoff"),
            spec_dens=params.get_param("spec_dens"),
            only_positive=only_positive_particle
        )
        if not only_positive_particle:
            #hole component (beta -> -beta)
            cont_integral_hole = cf.cont_integral( 
                t=times,
                beta= - params.get_param("beta"),
                upper_cutoff=params.get_param("upper_cutoff"),
                spec_dens=params.get_param("spec_dens"),
                only_positive=False
            )


        for tau, N_max in enumerate(N_maxs):
            params.update_parameters({"N_max": N_max})

            if only_positive_particle:
                cont_integral = cont_integral_particle[:N_max]
            else:
                #join the two arrays for the particle and hole components
                cont_integral = np.concatenate((cont_integral_particle[:N_max], cont_integral_hole[:N_max]))
        
            for h, h_val in enumerate(h_vals):
                params.update_parameters(
                    {"h": h_val}
                )  # this automatically updates "m" and "n" to reach to discrete cutoffs defined in class 'KernelParams'.

                # Create DiscrError object which holds the error w.r.t. to the continous results, and all associated parameters.
                discr_error = DiscrError(
                    **params.params, cont_integral_init=cont_integral, only_positive_particle=only_positive_particle
                )

                print("BEFORE: max. and min. freq: ", discr_error.fine_grid[-1], discr_error.fine_grid[0])
                if optimize:
                    discr_error.optimize(
                        rel_error_diff=rel_error_diff
                    )  # optimize values for m and n

                print("AFTER: max. and min. freq: ", discr_error.fine_grid[-1], discr_error.fine_grid[0])

                # create DecompKernel object which holds the kernel matrix and all associated parameters.
                # Note: big data attributes are not copied but passed as references to the original object, avoiding memory duplication.
                decomp_kernel = DecompKernel(discr_error)

                # compute reconstruction error (between reconstructed propagator and continuous-frequency propagator)
                propagator_reconstr = decomp_kernel.reconstruct_propagator_ID()

                # compute error between reconstructed and continuous-frequency propagator
                error_reconstr_vs_cont = discr_error.error_time_integrated(
                    time_series_approx=propagator_reconstr
                )
                # compute error between reconstructed and discrete propagator
                error_reconstr_vs_discr = discr_error.error_time_integrated(
                    time_series_exact=propagator_reconstr
                )

                # store error data in dictionary whose content will be added to hdf5 file.
                ID_propagator_error = {
                    "ID_error_reconstr_vs_cont": error_reconstr_vs_cont,
                    "ID_error_reconstr_vs_discr": error_reconstr_vs_discr,
                }

                # store to hdf5 file
                h5_kernel.append_kernel_element(
                    (h, tau, b),
                    kernel_object=decomp_kernel,
                    kernel_object2=discr_error,
                    dict_data=ID_propagator_error,
                )


def compute_AAA_grid_and_store(error_tolerances,
    N_maxs,
    betas,
    params: KernelParams,
    h5_kernel: Hdf5Kernel,
    remove_Froissart: bool = True,
) -> None:
    """
    This function computes the error and the corresponding AAAKernel object for each point on a data grid

    Parameters:
        error_tolerances (array type): Array of errors values which are used as convergence criteria for the AAA algorithm.
        N_maxs (array type): Array of total number of time steps values to be evaluated.
        betas (array type): Array of inverse temperature values to be evaluated.
        params (KernelParams): An instance of KernelParams that holds the parameter set.
        h5_kernel (Hdf5Kernel): An instance of Hdf5Kernel associated with the HDF5 file for storing the results.
        remove_Froissart (bool, optional): If True, Froissart doublets are removed.
    
    Returns:
        None
    """
    for b, beta in enumerate(betas):
        params.update_parameters({"beta": beta})


        # set time grid for maximal time needed
        times = cf.set_time_grid(
            N_max=N_maxs[-1], delta_t=params.get_param("delta_t")
        )
        # compute continuous-frequency integral such that they are not recomputed for every value of h below
        print("computing continuous integral..")
        #particle component
        cont_integral_particle = cf.cont_integral(
            t=times,
            beta=params.get_param("beta"),
            upper_cutoff=params.get_param("upper_cutoff"),
            spec_dens=params.get_param("spec_dens"),
            phi = 0, #along real axis
        )
        #hole component (beta -> -beta)
        cont_integral_hole = cf.cont_integral( 
            t=times,
            beta= - params.get_param("beta"),
            upper_cutoff=params.get_param("upper_cutoff"),
            spec_dens=params.get_param("spec_dens"),
            phi = 0, #along real axis
        )
        print("..finished")

        #disc_error = DiscrError(
        #    **params.params, cont_integral_init = np.concatenate((cont_integral_particle, cont_integral_hole))
        #)
        #m_init, n_init = params.params["m"], params.params["n"]
        #print("Optimizing m,n. Initial values: m = ", params.params["m"], "n= ", params.params["n"], "h= ", params.params["h"])
        #optimize values for m and n
        #disc_error.optimize(update_params = params, rel_error_diff=0.01)

        #make m and n large to include some zeros
        #params.update_parameters({"m":  int(np.min([params.params["m"] * 20, m_init])), "n": int(np.min([params.params["n"] * 20, n_init]))})
        #print("optimized m,n", params.params["m"], params.params["n"], ". Maximal frequency: ", disc_error.fine_grid[-1], ". Minimal frequency: ", disc_error.fine_grid[0])
    

        for tol_iter, tol in enumerate(error_tolerances):

            print("computing AAA")
            # Create DiscrError object which holds the error w.r.t. to the continous results, and all associated parameters.
            AAA_kernel = AAAKernel(**params.params, tol = tol)

            print(AAA_kernel.Z)
            print(AAA_kernel.F_particle)
            print(AAA_kernel.F_hole)
            print(AAA_kernel.nbr_poles_upper)
            
            if remove_Froissart:
                AAA_kernel.remove_Froissart()#remove Froissart doublets
                    
            for tau, N_max in enumerate(N_maxs):
            
                #join the two arrays for the particle and hole components
                cont_integral = np.concatenate((cont_integral_particle[:N_max], cont_integral_hole[:N_max]))
                
                #compute the propagator as given by the AAA algorithm
                propagator_AAA = AAA_kernel.propagator_AAA(times[:N_max])

                print("AAA", propagator_AAA[:10])
                print("cont", cont_integral[:10])

                # compute error between reconstructed and continuous-frequency propagator
                error_reconstr_vs_cont = cf.error_time_integrated(
                    time_series_exact = cont_integral,
                    time_series_approx = propagator_AAA,
                    delta_t = params.get_param("delta_t")
                )

                # store error data in dictionary whose content will be added to hdf5 file.
                AAA_dict = {
                    "AAA_error_reconstr_vs_cont": error_reconstr_vs_cont, "delta_t": params.get_param("delta_t"), "N_max": N_max
                }

                print(f"AAA error = {AAA_dict["AAA_error_reconstr_vs_cont"]} for parameters: tolerance = {tol}, m = {params.params["m"]}, n= {params.params["n"]}, beta = {beta}, h = {params.params["h"]}")

                # store to hdf5 file
                h5_kernel.append_kernel_element(
                    (tol_iter, tau, b),
                    kernel_object=AAA_kernel,
                    dict_data=AAA_dict,
                )

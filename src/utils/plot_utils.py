import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from src.data_interpolation.data_interpolation import DataInterp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import src.utils.common_funcs as cf
from src.store_kernel.store_kernel import Hdf5Kernel



# Define the logarithmic function to fit
def log_func(x, a, b):
    return a * np.log(x) + b


def fit_logarithmic(x_data, y_data):
    """
    Fit a logarithmic curve to the given data points.

    :param x_data: array-like, independent data
    :param y_data: array-like, dependent data
    :return: parameters of the fitted logarithmic function
    """

    # Use curve_fit to find the best fit parameters
    params, _ = curve_fit(log_func, x_data, y_data)

    return params

def plot_modes_vs_eps_and_h(kernel_dims, data_h5: dict) -> (plt.Figure, plt.Axes):
    """
    Function to plot the number of modes versus the error and the discretization parameter h.
    
    Params:
        kernel_dims (tuple): Dimensions of the parameter grid.
        data_h5 (dict): Dictionary holding the data from the hdf5 file.
    Returns:
        matplotlib figure object for further processing or saving
    """

    errors = data_h5["eps"]
    m_vals = data_h5["m"]
    n_vals = data_h5["n"]
    h_vals = data_h5["h"]
    N_maxs = data_h5["N_max"]
    betas = data_h5["beta"]
    delta_t_vals = data_h5["delta_t"]
    ID_ranks = data_h5["ID_rank"]

    nbr_modes_total = m_vals + n_vals
   
    #_____Plot the number of modes versus the error and the discretization parameter h____
    fig, axs = plt.subplots(kernel_dims[2], 2, figsize=(10.5, 2 * kernel_dims[2]))
    fig.suptitle(f'NUMBER OF BARE MODES AND ID-RANK VS. DISCRETIZATION PARAMETER $h$ AND ERROR $\epsilon$.')

    cmaps = [plt.get_cmap("Blues"), plt.get_cmap("Reds")]
    colors_blue = cmaps[0](np.linspace(0.35, 0.9, kernel_dims[1]))
    colors_orange = cmaps[1](np.linspace(0.35, 0.9, kernel_dims[1]))

    for b in range (kernel_dims[2]):
        for tau in range (kernel_dims[1]):
            mask = errors[:,tau,b] > 1.e-12#mask to filter values that are zero (i.e. no value avaliable)
            
            axs[b,0].plot(h_vals[mask, tau, b],nbr_modes_total[mask,tau,b],color=colors_orange[tau],marker="o",linestyle="-",markersize=3, label = r'$t={}$'.format(int(np.max(N_maxs[:,tau,:] * delta_t_vals[:,tau,:]))))
            axs[b,0].plot(h_vals[mask, tau, b], ID_ranks[mask,tau,b], color=colors_blue[tau], marker="o", linestyle="-", markersize=3)

            axs[b,1].plot(errors[mask, tau, b],nbr_modes_total[mask,tau,b] ,color=colors_orange[tau],marker="o",linestyle="-",markersize=3)
            axs[b,1].plot(errors[mask, tau, b], ID_ranks[mask,tau,b], color=colors_blue[tau], marker="o", linestyle="-", markersize=3, label = r'$t={}$'.format(int(np.max(N_maxs[:,tau,:] * delta_t_vals[:,tau,:]))))


        axs[b, 0].set_title(r'$\beta = {}$'.format(betas[0,0,b]), loc='right', x=.8, y=0.7)
        axs[b, 1].set_title(r'$\beta = {}$'.format(betas[0,0,b]), loc='right', x=.8, y=0.7)

        axs[b,0].set_yscale("log")   
        axs[b,0].set_xscale("log")  
        axs[b,1].set_xscale("log")

    axs[-1,0].set_xlabel(r"$h$")
    axs[-1,1].set_xlabel(r"$\epsilon$")

    # Create legends and place them outside the rightmost subplot
    legend1 = axs[0,0].legend(title=f'Nbr. bare modes,\n $m+n$', loc='upper left', bbox_to_anchor=(2.22, 1), borderaxespad=0.)
    legend2 = axs[0,1].legend(title=f'ID-rank', loc='upper left', bbox_to_anchor=(1.3, 1), borderaxespad=0.)


    #return matplotlib figure object for further processing or saving
    #save the figure from the output using "fig.savefig('filename.pdf', bbox_inches='tight')"
    return fig, axs


def plot_modes_vs_final_time_fixed_eps(kernel_dims, data_h5: dict) -> (plt.Figure, plt.Axes):
    """
    Function to plot the number of modes versus the final time t for fixed error eps.

    Params:
        kernel_dims (tuple): Dimensions of the parameter grid.
        data_h5 (dict): Dictionary holding the data from the hdf5 file.
    Returns:
        matplotlib figure object for further processing or saving
    """

    errors = data_h5["eps"]
    m_vals = data_h5["m"]
    n_vals = data_h5["n"]
    N_maxs = data_h5["N_max"]
    betas = data_h5["beta"]
    delta_t_vals = data_h5["delta_t"]
    ID_ranks = data_h5["ID_rank"]

    nbr_mode_vals = m_vals + n_vals

    cmap = plt.get_cmap("Blues_r")
    cmap_reds = plt.get_cmap("Reds_r")
    # Configuration and data preparation
    error_grid = np.logspace(-10, -3, 7)
    colors = cmap(np.linspace(0.35, 0.9, len(error_grid)))

    fig, axs = plt.subplots(kernel_dims[2], 4, figsize=(15.5, 2 * kernel_dims[2]))
    fig.suptitle(f'NUMBER OF MODES VS. FINAL TIME $t$.\n $m$: Nbr. of modes for frequencies $\omega > 1/e$,\n $n$: Nbr. of modes for frequencies $\omega < 1/e.$')

        

    nbr_mode_vals_interp = np.zeros((len(error_grid), kernel_dims[1]))
    
    ID_ranks_interp = np.zeros((len(error_grid), kernel_dims[1]))

    t_data = np.array([np.max(N_maxs[:,tau,:] * delta_t_vals[:,tau,:]) for tau in range (kernel_dims[1])])


    # Use logarithmic normalization
    norm = mcolors.LogNorm(vmin=np.min(error_grid[error_grid > 0]), vmax=10*np.max(error_grid))



    # Interpolation and plotting
    for b in range (kernel_dims[2]):

        ID_fitting_params_a = np.zeros((len(error_grid),))
        nbr_mode_fitting_params_a = np.zeros((len(error_grid),))

        for tau in range (kernel_dims[1]):
            mask = errors[:,tau,b] > 0. # Mask to filter values clearly above machine precision
            NBR_MODE_VALS_INTERP = DataInterp(errors[mask, tau,b], nbr_mode_vals[mask, tau,b])
            ID_RANKS_INTERP = DataInterp(errors[mask, tau,b], ID_ranks[mask, tau,b])

            nbr_mode_vals_interp[:, tau] = NBR_MODE_VALS_INTERP.interp(error_grid, x_scale='log')

            ID_ranks_interp[:, tau] = ID_RANKS_INTERP.interp(error_grid, x_scale='log')
        
        
        for eps_iter, eps_val in enumerate(error_grid):
            label = f'${eps_val:.1e}$'  # Scientific notation for error values
            #plot only nonzero values
            mask_nbr_mode = nbr_mode_vals_interp[eps_iter, :] > 0.

            axs[b,0].plot(t_data[mask_nbr_mode], nbr_mode_vals_interp[eps_iter, mask_nbr_mode], marker="o", linestyle="-", markersize=3, color=cmap_reds(norm(eps_val)), label=label)
            axs[b,2].plot(t_data[mask_nbr_mode], ID_ranks_interp[eps_iter, mask_nbr_mode], marker="o", linestyle="-", markersize=3, color=cmap(norm(eps_val)), label=label)
            
            #fit logarithmic curves
            params_nbr_mode_fit = fit_logarithmic(t_data[mask_nbr_mode][:], nbr_mode_vals_interp[eps_iter, mask_nbr_mode][:])
            nbr_mode_fitting_params_a[eps_iter] = params_nbr_mode_fit[0]

            params_ID_fit = fit_logarithmic(t_data[mask_nbr_mode][3:], ID_ranks_interp[eps_iter, mask_nbr_mode][3:])
            ID_fitting_params_a[eps_iter] = params_ID_fit[0]

            axs[b,2].plot(t_data[mask_nbr_mode], log_func(t_data[mask_nbr_mode], *params_ID_fit),  color=cmap(norm(eps_val)), linestyle = 'dotted')
            axs[b,0].plot(t_data[mask_nbr_mode], log_func(t_data[mask_nbr_mode], *params_nbr_mode_fit),  color=cmap_reds(norm(eps_val)), linestyle = 'dotted')

        
            #plot fitting parameters
            axs[b,3].scatter(eps_val, params_ID_fit[0], marker="o",  color=cmap(norm(eps_val)), label = r'$a$')
            axs[b,1].scatter(eps_val, params_nbr_mode_fit[0], marker="o",  color=cmap_reds(norm(eps_val)), label = r'$a$')
         
        ID_a_fitting_params= fit_logarithmic(error_grid, ID_fitting_params_a)
        nbr_mode_a_fitting_params= fit_logarithmic(error_grid, nbr_mode_fitting_params_a)

        axs[b,3].plot(error_grid, log_func(error_grid,*ID_a_fitting_params), color = 'black', linestyle = 'dashed')
        #axs[b,1].plot(error_grid, log_func(error_grid,*nbr_mode_a_fitting_params), color = 'black', linestyle = 'dashed')

        axs[b, 0].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='right', x = .4, y=0.85)
        axs[b, 1].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='right', x = .4, y=0.85)
        axs[b, 2].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='right', x = .4, y=0.85)
        axs[b, 2].set_title("Fitting function: \n" + r'$y = a \log (t) + b$', loc='right', x = .7, y=0.7)
        axs[b, 0].set_title("Fitting function: \n" + r'$y = a \log (t) + b$', loc='right', x = .7, y=0.7)
        axs[b, 3].set_title("Fitting function: \n" + r'$y = {} \log (\epsilon) {}$'.format(*np.round(ID_a_fitting_params,2)), loc='right', x = .9, y=0.7)
        #axs[b, 1].set_title("Fitting function: \n" + r'$y = {} \log (\epsilon) {}$'.format(*np.round(nbr_mode_a_fitting_params,2)), loc='right', x = .9, y=0.7)
        

        axs[b,0].set_ylabel(r'$m+ n$')
        axs[b,1].set_ylabel("fitting parameter " + r'$a$')
        axs[b,2].set_ylabel('ID-rank')
        axs[b,3].set_ylabel("fitting parameter " + r'$a$')
    

        axs[b,0].set_xscale('log')
        axs[b,1].set_xscale('log')
        axs[b,1].set_ylim(bottom=0)
        axs[b,2].set_xscale('log')
        

        axs[b,3].set_xscale('log')
        axs[b,3].set_ylim(bottom=0)


        
    # Setting labels and legend
    axs[-1,0].set_xlabel(r'$t$')
    axs[-1,1].set_xlabel(r'$t$')
    axs[-1,2].set_xlabel(r'$t$')
    axs[-1,3].set_xlabel(r'$\epsilon$')




    # Add a colorbar to the figure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is necessary because we're not using sm with imshow or similar
    cbar = plt.colorbar(sm, ax=axs)
    cbar.set_label('Error '+r'$\epsilon$')

   
    #Return matplotlib figure object for further processing or saving
    #save the figure from the output using "fig.savefig('filename.pdf', bbox_inches='tight')"
    return fig, axs


def plot_modes_vs_final_time_fixed_h(kernel_dims, data_h5: dict) -> (plt.Figure, plt.Axes):
    """
    Function to plot the number of modes versus the final time t for discretization parameter h.

    Params:
        kernel_dims (tuple): Dimensions of the parameter grid.
        data_h5 (dict): Dictionary holding the data from the hdf5 file.
    Returns:
        matplotlib figure object for further processing or saving
    """

    m_vals = data_h5["m"]
    n_vals = data_h5["n"]
    h_vals = data_h5["h"]
    N_maxs = data_h5["N_max"]
    betas = data_h5["beta"]
    delta_t_vals = data_h5["delta_t"]
    ID_ranks = data_h5["ID_rank"]

    cmap = plt.get_cmap("Blues_r")
    cmap_reds = plt.get_cmap("Reds_r")
    # Configuration and data preparation
    error_grid = np.logspace(-10, -3, 7)
    colors = cmap(np.linspace(0.35, 0.9, len(error_grid)))

    fig, axs = plt.subplots(kernel_dims[2], 3, figsize=(15.5,2 * kernel_dims[2]))
    fig.suptitle(f'NUMBER OF MODES VS. FINAL TIME $t$.\n $m$: Nbr. of modes for frequencies $\omega > 1/e$,\n $n$: Nbr. of modes for frequencies $\omega < 1/e.$')

        

    #elementwise multiplication
    t_data = N_maxs * delta_t_vals


    # Use logarithmic normalization
    norm = mcolors.LogNorm(vmin=np.min(h_vals[h_vals > 0]), vmax=10*np.max(h_vals))



    # Interpolation and plotting
    for b in range (kernel_dims[2]):

        for tau in range (kernel_dims[1]):
        
            for h,h_val in enumerate(h_vals[:,tau, b]):

                if h_val > 0:
                    mask = t_data[h, : , b ] > 0
                    axs[b,0].plot(t_data[h, mask , b], m_vals[h, mask , b], marker="o", linestyle="-", markersize=3, color=cmap(norm(h_val)))
                    axs[b,1].plot(t_data[h, mask , b], n_vals[h, mask , b], marker="o", linestyle="-", markersize=3, color=cmap(norm(h_val)))
                    axs[b,2].plot(t_data[h, mask , b], ID_ranks[h, mask , b], marker="o", linestyle="-", markersize=3, color=cmap(norm(h_val)))
                
                else:
                    continue
                
        
        axs[b, 0].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='right', x = .4, y=0.85)
        axs[b, 1].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='right', x = .4, y=0.85)
        axs[b, 2].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='right', x = .4, y=0.85)
   
        axs[b,0].set_ylabel(r'$m$')
        axs[b,1].set_ylabel(r'$n$')
        axs[b,2].set_ylabel('ID-rank')
  

        axs[b,0].set_xscale('log')
        axs[b,1].set_xscale('log')
        axs[b,2].set_xscale('log')

  

        
    # Setting labels and legend
    axs[-1,0].set_xlabel(r'$t$')
    axs[-1,1].set_xlabel(r'$t$')
    axs[-1,2].set_xlabel(r'$t$')

    # Add a colorbar to the figure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is necessary because we're not using sm with imshow or similar
    cbar = plt.colorbar(sm, ax=axs)
    cbar.set_label('Discr. parameter '+r'$h$')

    return fig, axs




def plot_h_vs_eps_fixed_time(kernel_dims, data_h5: dict) -> (plt.Figure, plt.Axes):
    """
    Function to plot the discretization parameter h versus the error eps for fixed final time t.
    Params:
        kernel_dims (tuple): Dimensions of the parameter grid.
        data_h5 (dict): Dictionary holding the data from the hdf5 file.
    Returns:
        matplotlib figure object for further processing or saving
    """

    errors = data_h5["eps"]
    h_vals = data_h5["h"]
    N_maxs = data_h5["N_max"]
    betas = data_h5["beta"]
    delta_t_vals = data_h5["delta_t"]

    cmap = plt.get_cmap("Reds")

    #elementwise multiplication
    t_data = N_maxs * delta_t_vals

    # Use logarithmic normalization
    norm = mcolors.LogNorm(vmin=np.min(t_data[t_data > 0]), vmax=10*np.max(t_data))

    fig, axs = plt.subplots(kernel_dims[2], figsize=(5.5, 2*kernel_dims[2]), sharex= True)

    for b in range (kernel_dims[2]):
        beta = np.max(betas[:,:,b])
        for tau in range (kernel_dims[1]):
            t = np.max(t_data[:,tau,:])
            axs[b].plot(1 / np.log(1 / errors[:, tau, b]) , h_vals[:, tau, b], color = cmap(norm(t)))

            axs[b].set_ylabel(r'$h$')
            axs[b,].set_title(f'beta = {beta}', loc='right', x = .7, y=0.7)
            axs[b].set_ylim(bottom = 0)

    axs[-1].set_xlabel(r"$1/\log(\epsilon_{discr}^{-1})$")
    axs[-1].set_xlim(left = 0,right = .4)

    # Add a colorbar to the figure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is necessary because we're not using sm with imshow or similar
    cbar = plt.colorbar(sm, ax=axs)
    cbar.set_label('Final time '+r'$t$')

    #Return matplotlib figure object for further processing or saving
    return fig, axs


def plot_error_vs_final_time_fixed_h(kernel_dims, data_h5: dict) -> (plt.Figure, plt.Axes):

    """ 
    Function to plot the error eps versus the final time t for fixed discretization parameter h.
    Params:
        kernel_dims (tuple): Dimensions of the parameter grid.
        data_h5 (dict): Dictionary holding the data from the hdf5 file.
    Returns:
        matplotlib figure object for further processing or saving
    """

    errors = data_h5["eps"]
    h_vals = data_h5["h"]
    N_maxs = data_h5["N_max"]
    betas = data_h5["beta"]
    delta_t_vals = data_h5["delta_t"]

    # Choose a colormap
    cmap = plt.get_cmap('Blues_r')  

    fig, axs = plt.subplots(kernel_dims[2], 1, figsize=(5.5, 8.5), sharex=True)
    fig.suptitle(f'ERROR $\epsilon$ VS. FINAL TIME $t$')


    t_data = np.array([np.max(N_maxs[:,tau,:] * delta_t_vals[:,tau,:]) for tau in range (kernel_dims[1])])


    start_index = 0
    h_vals_loop = np.array([np.max(h_vals[h,:,:]) for h in range (kernel_dims[0])])[start_index:]


    # Normalize h_val values to the 0-1 range
    norm = mcolors.Normalize(vmin=np.min(h_vals_loop[h_vals_loop > 0.]), vmax=1.3*np.max(h_vals_loop))

    for b in range (kernel_dims[2]):
        
        for h_idx, h_val in enumerate(h_vals_loop):
            
            mask = errors[h_idx + start_index,:,b] > 1.e-16
            axs[b].plot(t_data[mask], errors[h_idx + start_index,mask,b], color = cmap(norm(h_val)), marker="o",linestyle="-",markersize=3)
        

        axs[b].set_title(r'$\beta = {}$'.format(np.max(betas[:,:,b])), loc='left', x=.3, y=.6)
        
        axs[b].set_ylabel(r'$\epsilon$')
        axs[b].set_yscale('log')

    axs[-1].set_xlabel(r'$t$')
    axs[-1].set_xscale('log')



    # Add a colorbar to the figure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is necessary because we're not using sm with imshow or similar
    cbar = plt.colorbar(sm, ax=axs)
    cbar.set_label('Discretization parameter '+r'$h$')

    return fig, axs


def plot_frequency_grids(h5_kernel: Hdf5Kernel) -> (plt.Figure, plt.Axes):
    """
    Function to plot the frequency grids for fixed final time and fixed beta.
    Params:
        kernel_dims (tuple): Dimensions of the parameter grid.
        h5_kernel (Hdf5Kernel): Hdf5Kernel object associated with the HDF5 file for storing the results.
    Returns:
        matplotlib figure object for further processing or saving
    """
    kernel_dims = h5_kernel.kernel_dims

    nbr_h_values = 12
    nbr_taus = 5
    # Choose a colormap
    fig, axs = plt.subplots(nbr_taus, 2, figsize=(14.5, 1.3* kernel_dims[1]), sharex= True)
    fig.suptitle(f'Frequency grids')
    cmap_blues = plt.get_cmap('Blues_r') 
    cmap_reds = plt.get_cmap('Reds_r') 
    cmap_greens = plt.get_cmap('Greens_r') 


    for tau in range (nbr_taus):
        #extract params and data for fixed final time and fixed beta
        params_array, data_array = np.array(list(zip(*[h5_kernel.read_kernel_element((kernel_dims[0]-1-i, kernel_dims[1]-1 - tau, kernel_dims[2]-1)) for i in range (nbr_h_values)])))

        h_vals = [params["h"] for params in params_array]
        fine_grids =  [data["fine_grid"] for data in data_array]
        coarse_grids =  [data["coarse_grid"] for data in data_array]

        min_h_val = np.min(h_vals)
        max_h_val = np.max(h_vals)

        # Normalize h_val values to the 0-1 range
        norm = mcolors.Normalize(vmin=min_h_val, vmax=1.2*max_h_val)

        lower_lim, upper_lim, interval_spacing = -14, 6, "log"


        for i, h_val in enumerate(h_vals):
            
            if h_val > 0:
                density_fine, density_fine_grid = cf.point_density(fine_grids[i], lower_lim, upper_lim, interval_spacing)
                density_coarse, density_coarse_grid = cf.point_density(coarse_grids[i], lower_lim, upper_lim, interval_spacing)
                density_diff, density_diff_grid = cf.point_density(np.setdiff1d(fine_grids[i], coarse_grids[i]), lower_lim, upper_lim, interval_spacing)


                axs[tau,0].plot(density_fine_grid, density_fine, marker = ".", linestyle = 'dashed', markersize = 3, color = cmap_reds(norm(h_val)))
                axs[tau,0].plot(density_coarse_grid, density_coarse, marker = ".", markersize = 3, color = cmap_greens(norm(h_val)))
                axs[tau,1].plot(density_diff_grid, density_diff, marker = ".", markersize = 3, color = cmap_blues(norm(h_val)))
                    

            else:
                continue

        for ax in axs[tau,:]:
            ax.set_xscale("log")
            

        axs[tau,0].set_ylabel("Density of frequency points")

        axs[tau,0].set_title(f'Fine vs. ID Grids,\n t={params_array[-1]["delta_t"] * params_array[-1]["N_max"]},\n beta = {params_array[-1]["beta"]}', loc='right', x=.5, y=0.5)
        axs[tau,1].set_title(f'Differences,\n t={params_array[-1]["delta_t"] * params_array[-1]["N_max"]},\n beta = {params_array[-1]["beta"]}', loc='right', x=.3, y=0.5)

    axs[-1,0].set_xlabel(r"$\omega_k$")
    axs[-1,1].set_xlabel(r"$\omega_k$")

    # Add a colorbar to the figure
    sm = cm.ScalarMappable(cmap=cmap_blues, norm=norm)
    sm.set_array([])  # This is necessary because we're not using sm with imshow or similar
    cbar = plt.colorbar(sm, ax=axs)
    cbar.set_label('Discretization parameter '+r'$h$')


    return fig, axs

def plot_h_vs_eps(kernel_dims, data_h5: dict) -> (plt.Figure, plt.Axes):
    """
    Function to plot the discretization parameter h versus the error eps for fixed final time t.
    """
    errors = data_h5["eps"]
    h_vals = data_h5["h"]
    N_maxs = data_h5["N_max"]

    # Choose a colormap
    fig, axs = plt.subplots(1,1, figsize=(8, 5))
    cmap_blues = plt.get_cmap('Blues_r') 
    # Normalize h_val values to the 0-1 range
    norm = mcolors.Normalize(vmin=N_maxs[0,0,0], vmax=1.2 * N_maxs[0,-1,0])

    for tau in range (kernel_dims[1]):
        N_max = N_maxs[0,tau,0]
        mask = errors[:,tau,0] > 1.e-15
        axs.plot(1 / np.log(1 / errors[mask, tau, 0]), h_vals[mask, tau, 0], marker="o", linestyle="-", markersize=3, color = cmap_blues(norm(N_max)), label = r"${}$".format(data_h5["delta_t"][0,tau,0]*N_max ))

    #add legend
    axs.legend(title = r'$t$')
    axs.set_xlabel(r"$1/\log(\epsilon_{discr}^{-1})$")
    axs.set_ylabel(r"$h(\epsilon_{discr})$")


    return fig, axs
from src.AAA.baryrat import aaa
import numpy as np
from src.kernel_params.kernel_params import KernelParams
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.spec_dens.spec_dens import spec_dens_gapless
from src.utils import common_funcs as cf


#set up parameters for specified spectral densities
params = KernelParams(spec_dens = lambda omega: np.sqrt(1 + omega**2), N_max = 1000)

#create KernelMatrix object with specified parameters which we will use as reference 
K = KernelMatrix(**params.params)

#explicitly get the fine frequency grid
fine_grid, k_values, Jacobian = K._initialize_fine_grid()

#consider rotated contour in complex plane
Z = fine_grid* np.exp(1.0j * K.phi) # frequency points in complex plane
F = K.kernel[0,:] # function values on the rotated contour, determined by spectral density and Fermi Dirac distribution

#perform AAA algorithm on spectral density multiplied with Fermi-Dirac distribution
r, errors = aaa(Z = Z, F = F, return_errors=True)
poles, residues = r.polres()
print("Poles", poles)
print("Residues", residues)
print("Errors", errors)
import numpy as np

MAX_EXP_ARG = np.log(np.finfo(np.float64).max)

# define a collection of vectorized spectral densities
def spec_dens_gapless(omega, cutoff_lower: float = -100., cutoff_upper: float = 100.0, Gamma: float = 1.0) -> float: 
        """
        Gapless spectral density with smooth cutoffs.
        Sharpness of cutoff is controlled by 'sharpness'.

        Vectorized implementation.
        

        Parameters:
        - cutoff_lower (float, optional): Lower cutoff frequency. Should be negative.
        - cutoff_upper (float, optional): Upper cutoff frequency. Should be positive.
        - Gamma (float, optional): Energy scale of the spectral density.
   
        Returns:
        scalar/np.ndarray: Spectral density evaluated at the specified frequency values.
        """ 

        sharpness = 10 * Gamma
        
        safe_omega_upper = np.clip(sharpness * (omega - cutoff_upper), None,  MAX_EXP_ARG)
        safe_omega_lower = np.clip(- sharpness * (omega - cutoff_lower), None, MAX_EXP_ARG)

        denominator = (1 + np.exp(safe_omega_lower)) * (1 + np.exp(safe_omega_upper))
       
        val = Gamma / denominator
        
        return val
import numpy as np

MAX_EXP_ARG = np.log(np.finfo(np.float64).max)
MAX_FLOAT = np.finfo(np.float64).max

# define a collection of vectorized spectral densities
def spec_dens_gapless(
    omega: np.ndarray,
    cutoff_lower: float = -10.0,
    cutoff_upper: float = 10.0,
    Gamma: float = 1.0,
) -> np.ndarray:
    """
    Gapless spectral density with smooth cutoffs.
    Sharpness of cutoff is controlled by 'sharpness'.

    Vectorized implementation.


    Parameters:
    - omega (scalar/np.ndarray): Frequency values at which the spectral density is evaluated.
    - cutoff_lower (float, optional): Lower cutoff frequency.
    - cutoff_upper (float, optional): Upper cutoff frequency.
    - Gamma (float, optional): Energy scale of the spectral density.

    Returns:
    scalar/np.ndarray: Spectral density evaluated at the specified frequency values.
    """
    #hard code sharpness (chosen by hand)
    sharpness = 10 * Gamma

    safe_omega_upper = np.clip(sharpness * (omega - cutoff_upper), None, MAX_EXP_ARG/2 )
    safe_omega_lower = np.clip(-sharpness * (omega - cutoff_lower), None, MAX_EXP_ARG/2 )

    denominator = (1 + np.exp(safe_omega_lower)) * (1 + np.exp(safe_omega_upper))

    val = Gamma / denominator

    return val


def spec_dens_gapped_sym(
    omega: np.ndarray,
    cutoff_lower: float = 2.0,
    cutoff_upper: float = 10.0,
    Gamma: float = 1.0,
) -> np.ndarray:
    """
    Gapped spectral density with smooth cutoffs. Symmetric around 0.
    Parameters specify the spectral density for positive frequencies.

    Parameters:
    - cutoff_lower (float, optional): Lower cutoff frequency for positive frequencies. Should be positive.
    - cutoff_upper (float, optional): Upper cutoff frequency for positive frequencies. Should be positive.
    - Gamma (float, optional): Energy scale of the spectral density.

    Returns:
    scalar/np.ndarray: Spectral density evaluated at the specified frequency values.
    """

    val = spec_dens_gapless(
        omega, cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, Gamma=Gamma
    ) + spec_dens_gapless(
        omega, cutoff_lower=-cutoff_upper, cutoff_upper=-cutoff_lower, Gamma=Gamma
    )

    return val


if __name__ == "__main__":
 
    # plot spectral densities
    import matplotlib.pyplot as plt

    x = np.linspace(0,40, 1000) * np.exp(1j * np.pi / 4)
    #plt.plot(x, np.real(spec_dens_gapless(x)), linewidth = 3, label="gapless", color = 'blue')
    #plt.plot(x, np.imag(spec_dens_gapless(x)), linewidth = 3, alpha = 0.5, color = 'blue')
    plt.plot(x, np.real(spec_dens_gapped_sym(x, cutoff_lower=10, cutoff_upper=1.e6)), linewidth = 1.5, linestyle = 'dashed', label="gapped", color = 'red')
    plt.plot(x, np.imag(spec_dens_gapped_sym(x, cutoff_lower=10, cutoff_upper=1.e6)), linewidth = 1.5, linestyle = 'dashed', alpha = 0.5, color = 'red')
    plt.legend(loc = "upper right")
    plt.xlabel("Frequency " + r"$\omega$")
    plt.ylabel("Spectral density " + r"$\Gamma(\omega)$")
    plt.show()

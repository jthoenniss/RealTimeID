import numpy as np

MAX_EXP_ARG = np.log(np.finfo(np.float64).max)
MAX_FLOAT = np.finfo(np.float64).max


# define a collection of vectorized spectral densities
def spec_dens_gapless(
    omega: np.ndarray,
    cutoff_lower: float = -1.0e6,
    cutoff_upper: float = 1.0e6,
    Gamma: float = 1.0,
    sharpness=None,
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
    # hard code sharpness (chosen by hand)
    if sharpness is None:
        sharpness = 10 * Gamma

    safe_omega_upper = np.clip(
        sharpness * (omega - cutoff_upper), None, MAX_EXP_ARG / 2
    )
    safe_omega_lower = np.clip(
        -sharpness * (omega - cutoff_lower), None, MAX_EXP_ARG / 2
    )

    denominator = (1 + np.exp(safe_omega_lower)) * (1 + np.exp(safe_omega_upper))

    val = Gamma / denominator

    return val


class SpecDensGapless:

    def __init__(
        self,
        Lambda: float = 1.0e6,
        Gamma: float = 1.0,
        sharpness: float = None,
    ):
        self.Lambda = Lambda
        self.Gamma = Gamma
        if sharpness is None:
            self.sharpness = 10 * Gamma
        else:
            self.sharpness = sharpness

    def __call__(self, omega: np.ndarray) -> np.ndarray:
        return spec_dens_gapless(
            omega,
            cutoff_lower=-self.Lambda,
            cutoff_upper=self.Lambda,
            Gamma=self.Gamma,
            sharpness=self.sharpness,
        )

    def _pole_freqs(self, omega: float, k: int):
        """
        Function to compute the poles frequencies.

        Parameters:
        - omega (float): Frequency value on real axis. Typically either Lambda or - Lambda.
        - k (int): Index of the pole.

        Returns:
        - float: Pole frequency.
        """
        return omega + 2.0j * np.pi * 1 / self.sharpness * (k + 0.5)

    def poles_residues(self, phi: float = np.pi / 4):
        """
        Compute the poles and residues below the integration contour with angle phi of the spectral density multiplied with the Fermi-Dirac distribution.

        Parameters:
        - phi (float, optional): Angle of the integration contour.

        Returns:
        - tuple: Poles and residues of the spectral density multiplied with the Fermi-Dirac distribution.
        """

        # define kappa in terms of sharpness and Lambda
        kappa = self.Lambda * self.sharpness / (2 * np.pi)

        # array that contains the poles frequencies
        poles_freq_table_pos = np.array(
            [
                self._pole_freqs(self.Lambda, k)
                for k in range(1 + np.floor(np.tan(phi) * kappa - 0.5).astype(int))
            ]
        )
        poles_freq_table_neg = np.array(
            [
                self._pole_freqs(-self.Lambda, k)
                for k in range(1 + np.floor(np.tan(phi) * kappa - 0.5).astype(int))
            ]
        )

        # residues of spectral density
        residue_spec_dens_pos = (
            -1
            / (1 + np.exp(-self.sharpness * (poles_freq_table_pos + self.Lambda)))
            * 1
            / self.sharpness
        )
        residue_spec_dens_neg = (
            1
            / (1 + np.exp(self.sharpness * (poles_freq_table_neg - self.Lambda)))
            * 1
            / self.sharpness
        )

        poles = np.concatenate((poles_freq_table_neg, poles_freq_table_pos))
        residues = np.concatenate((residue_spec_dens_neg, residue_spec_dens_pos))

        return poles, residues


def spec_dens_gapped_sym(
    omega: np.ndarray,
    cutoff_lower: float = 0.5,
    cutoff_upper: float = 1.0e6,
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


def spec_dens_exp(
    omega: np.ndarray, Gamma: float = 1.0, Lambda: float = 1.0e4
) -> np.ndarray:
    """
    Exponential spectral density.

    Parameters:
    - omega (scalar/np.ndarray): Frequency values at which the spectral density is evaluated.
    - Gamma (float, optional): Energy scale of the spectral density.
    - Lambda (float, optional): Cutoff frequency.

    Returns:
    scalar/np.ndarray: Spectral density evaluated at the specified frequency values.
    """

    val = Gamma * np.exp(-omega / Lambda)
    return val


def spec_dens_semi_circle(
    omega: np.ndarray, Gamma: float = 1.0, half_width: float = 1.0
) -> np.ndarray:
    """
    Semicircle spectral density.

    Parameters:
    - omega (scalar/np.ndarray): Frequency values at which the spectral density is evaluated.
    - Gamma (float, optional): Energy scale of the spectral density.
    - half_width (float, optional): Half-width of the semicircle.

    Returns:
    scalar/np.ndarray: Spectral density evaluated at the specified frequency values.
    """
    omega = np.asarray(omega)  # Ensure omega is a NumPy array
    result = np.zeros_like(
        omega
    )  # Initialize result array with the same shape as omega

    mask = (
        np.abs(omega) < half_width
    )  # Boolean mask for values where abs(omega) < half_width
    result[mask] = np.sqrt(
        half_width**2 - omega[mask] ** 2
    )  # Apply the formula only where the condition is True

    return Gamma * result


if __name__ == "__main__":

    # plot spectral densities
    import matplotlib.pyplot as plt

    x = np.linspace(0, 40, 1000) * np.exp(1j * np.pi / 4)
    # plt.plot(x, np.real(spec_dens_gapless(x)), linewidth = 3, label="gapless", color = 'blue')
    # plt.plot(x, np.imag(spec_dens_gapless(x)), linewidth = 3, alpha = 0.5, color = 'blue')
    plt.plot(
        x,
        np.real(spec_dens_gapped_sym(x, cutoff_lower=10, cutoff_upper=1.0e6)),
        linewidth=1.5,
        linestyle="dashed",
        label="gapped",
        color="red",
    )
    plt.plot(
        x,
        np.imag(spec_dens_gapped_sym(x, cutoff_lower=10, cutoff_upper=1.0e6)),
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
        color="red",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Frequency " + r"$\omega$")
    plt.ylabel("Spectral density " + r"$\Gamma(\omega)$")
    plt.show()

import numpy as np

MAX_EXP_ARG = np.log(np.finfo(np.float64).max)
MAX_FLOAT = np.finfo(np.float64).max


# define a collection of vectorized spectral densities
def spec_dens_gapless(
    omega: np.ndarray,
    sharpness: float,
    cutoff_lower: float = -1.0e6,
    cutoff_upper: float = 1.0e6,
    Gamma: float = 1.0,
) -> np.ndarray:
    """
    Gapless spectral density with smooth cutoffs.
    Sharpness of cutoff is controlled by 'sharpness'.

    Vectorized implementation.


    Parameters:
    - omega (scalar/np.ndarray): Frequency values at which the spectral density is evaluated.
    - sharpness (float): Determines the sharpness of the cutoff at Lambda.
    - cutoff_lower (float, optional): Lower cutoff frequency.
    - cutoff_upper (float, optional): Upper cutoff frequency.
    - Gamma (float, optional): Energy scale of the spectral density.

    Returns:
    scalar/np.ndarray: Spectral density evaluated at the specified frequency values.
    """
 
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
            self.sharpness = 20 / Lambda
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
    
    def _residue_pos(self, poles_pos):
        """
        Function to compute the residues of the spectral density multiplied with the Fermi-Dirac distribution.

        Parameters:
        - poles_pos (np.ndarray): Array of poles with positive real part.

        Returns:
        - np.ndarray: Residues of the spectral density multiplied with the Fermi-Dirac distribution.
        """
        return -1 / (1 + np.exp(-self.sharpness * (poles_pos + self.Lambda))) * 1 / self.sharpness
    
    def _residue_neg(self, poles_neg):
        """
        Function to compute the residues of the spectral density multiplied with the Fermi-Dirac distribution.

        Parameters:
        - poles_neg (np.ndarray): Array of poles with negative real part.

        Returns:
        - np.ndarray: Residues of the spectral density multiplied with the Fermi-Dirac distribution.
        """
        return 1 / (1 + np.exp(self.sharpness * (poles_neg - self.Lambda))) * 1 / self.sharpness

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
        residue_spec_dens_pos = self._residue_pos(poles_freq_table_pos)
        residue_spec_dens_neg = self._residue_neg(poles_freq_table_neg)
    

        poles = np.concatenate((poles_freq_table_neg, poles_freq_table_pos))
        residues = np.concatenate((residue_spec_dens_neg, residue_spec_dens_pos))

        return poles, residues

    def explicit_poles_and_residues(self, h: float, phi: float = np.pi/4, phi_low: float = np.pi/8, phi_up: float = 3*np.pi/8):
        """
        Compute the "explicit" poles and residues that should always be considered in discrete sum along complex contour when approximating continous intgegral along complex path.

        Parameters:
        - h(float): discretization parameter.
        - phi (float, optional): Angle of the integration contour.
        - phi_low (float, optional): Lower angle of the integration contour.
        - phi_up (float, optional): Upper angle of the integration contour.

        Returns:
        - tuple: Poles and residues of the spectral density multiplied with the Fermi-Dirac distribution.
        """

        def _explicit_poles(phi_low, phi_up, Lambda, sharpness):
            """
            Returns:
            - poles_pos: poles with positive real part
            - poles_neg: poles with negative real part
            """

            #summation limits
            limit_up = np.floor(1+np.tan(phi_up) * Lambda * sharpness / (2 * np.pi) - 1/2)
            limit_low = np.floor(1+np.tan(phi_low) * Lambda * sharpness / (2 * np.pi) - 1/2)

            #explicit poles between two rotation angles
            #positive real part of frequencies
            poles_pos = np.array([self._pole_freqs(omega = Lambda, k = k) for k in range(int(limit_low), int(limit_up))])
            #negative real part of frequencies
            poles_neg = np.array([self._pole_freqs(omega = -Lambda, k = k) for k in range(int(limit_low), int(limit_up))])

            return poles_pos, poles_neg
        
    
        add_poles_up_pos, add_poles_up_neg = _explicit_poles(phi_low = phi, phi_up = phi_up, Lambda = self.Lambda, sharpness = self.sharpness)
        add_poles_low_pos, add_poles_low_neg = _explicit_poles(phi_low = phi_low, phi_up = phi, Lambda = self.Lambda, sharpness = self.sharpness)

        #positive frequencies
        def upper_residues_pos(w, sharpness, h, Lambda, phi):
            return (1/sharpness) * 1/(1+np.exp(-sharpness * (w + Lambda)))* np.exp(2.j * np.pi* np.log(w * np.exp(-1.j * phi))/h)/(1-np.exp(2.j * np.pi  * np.log(w * np.exp(-1.j * phi))/h))
        def bottom_residues_pos(w, sharpness, h, Lambda, phi):
            return -(1/sharpness) * 1/(1+np.exp(-sharpness * (w + Lambda)))* np.exp(-2.j * np.pi * np.log(w * np.exp(-1.j * phi))/h)/(1-np.exp(-2.j * np.pi  * np.log(w * np.exp(-1.j * phi))/h))
        #negative frequencies
        def upper_residues_neg(w, sharpness, h, Lambda, phi):
            return -(1/sharpness) * 1/(1+np.exp(sharpness * (w - Lambda)))* np.exp(-2.j * np.pi  * np.log(-w * np.exp(1.j * phi))/h)/(1-np.exp(-2.j * np.pi  * np.log(-w * np.exp(1.j * phi))/h))
        def bottom_residues_neg(w, sharpness, h, Lambda, phi):
            return (1/sharpness) * 1/(1+np.exp(+sharpness * (w - Lambda)))* np.exp(2.j * np.pi  * np.log(-w * np.exp(1.j * phi))/h)/(1-np.exp(2.j * np.pi  * np.log(-w * np.exp(1.j * phi))/h))

        upper_res_pos_arr = upper_residues_pos(add_poles_up_pos, self.sharpness, h, self.Lambda, phi = phi)
        lower_res_pos_arr = bottom_residues_pos(add_poles_low_pos, self.sharpness, h, self.Lambda, phi = phi)
        upper_res_neg_arr = upper_residues_neg(add_poles_up_neg, self.sharpness, h, self.Lambda, phi = phi)
        lower_res_neg_arr = bottom_residues_neg(add_poles_low_neg, self.sharpness, h, self.Lambda, phi = phi)

        #concatenate poles and residues
        poles = np.concatenate((add_poles_up_pos, add_poles_low_pos, add_poles_up_neg, add_poles_low_neg))
        residues = np.concatenate((upper_res_pos_arr, lower_res_pos_arr, upper_res_neg_arr, lower_res_neg_arr))

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

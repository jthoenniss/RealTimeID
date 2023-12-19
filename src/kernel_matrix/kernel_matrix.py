import numpy as np
from src.utils import common_funcs as cf

class KernelMatrix:
    def __init__(
        self,
        m: int,
        n: int,
        beta: float,
        times: np.ndarray,
        h: float,
        phi: float,
    ):
        """
        Initialize parameters for creating a kernel matrix.

        Parameters:
        - m (int): Number of discretization intervals for omega > 1/e.
        - n (int): Number of discretization intervals for omega < 1/e.
        - beta (float): Inverse temperature.
        - times (np.ndarray): Array containing the points on the time grid.
        - h (float): Discretization parameter.
        - phi (float): Rotation angle in the complex plane.
        """
        self.m = m
        self.n = n
        self.beta = beta
        self.times = np.atleast_1d(times)[:, np.newaxis]  # Ensure times is at least 1D and reshape for broadcasting
        self.h = h
        self.phi = phi

    def set_fine_grid(self) -> np.ndarray:
        """
        Generates a fine grid for given discretization parameters.

        Returns:
        - np.ndarray: The generated fine grid as a NumPy array.
        - np.ndarray: The values for k that define the points on the grid.
        """
        k_values = np.arange(-self.n, self.m + 1)
        fine_grid = np.exp(self.h * k_values - np.exp(-self.h * k_values))
        return fine_grid, k_values

    def get_kernel(self) -> np.ndarray:
        """
        Creates the kernel matrix using the Fermi distribution function and spectral density.

        Returns:
        - np.ndarray: Kernel matrix.
        """

        fine_grid, k_values = self.set_fine_grid()
        fine_grid_complex = fine_grid * np.exp(1.0j * self.phi)

        # Kernel defined by Fermi distribution, multiplied by spectral density
        K = cf.distr(self.times, fine_grid_complex, self.beta) * cf.spec_dens_array(fine_grid_complex)
        K *= self.h * (1 + np.exp(-self.h * k_values)) * fine_grid_complex

        return K

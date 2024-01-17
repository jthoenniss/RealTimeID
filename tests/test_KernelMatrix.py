import unittest
import numpy as np
from src.kernel_matrix.kernel_matrix import KernelMatrix
from src.utils import common_funcs as cf
from src.spec_dens.spec_dens import spec_dens_gapless

class TestKernelMatrix(unittest.TestCase):
    def setUp(self):

        self.params_KernelMatrix = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
            "spec_dens":  lambda x: spec_dens_gapless(x)
        }
        self.K = KernelMatrix(**self.params_KernelMatrix)


    def test_all_attrs_present(self):
        KernelMatrix_keys_required = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel", "spec_dens", "spec_dens_array_cmplx"]

        KernelMatrix_keys = vars(self.K).keys()

        for key in KernelMatrix_keys_required:#check that all attributes exist.
            self.assertTrue(key in KernelMatrix_keys, f"Key {key} not found.")

    def test_initialization(self):
        # Test initialization of RtKernel
        self.assertEqual(self.K.m, 10)
        self.assertEqual(self.K.n, 5)
        self.assertEqual(self.K.beta, 1.0)
        self.assertEqual(self.K.N_max, 10)
        self.assertEqual(self.K.delta_t, 0.1)
        self.assertTrue(np.array_equal(self.K.times, cf.set_time_grid(N_max=10, delta_t=0.1)))
        self.assertEqual(self.K.h, 0.2)
        self.assertEqual(self.K.phi, np.pi / 4)
        self.assertEqual(self.K.spec_dens(0), 1.)

    def test_create_kernel(self):
        # Test the kernel creation method
        fine_grid, k_values = self.K.fine_grid, self.K.k_values
        k_values_check = np.arange(-self.K.n, self.K.m + 1)
        fine_grid_check = np.exp(
            self.K.h * k_values_check - np.exp(-self.K.h * k_values_check)
        )
        self.assertTrue(np.array_equal(k_values, k_values_check))
        self.assertTrue(np.array_equal(fine_grid, fine_grid_check))

    def test_kernel_matrix_shape(self):
        K = self.K.kernel
        expected_shape = (len(self.K.times), self.K.m + self.K.n + 1)
        self.assertEqual(K.shape, expected_shape)

    def test_spec_dens_array(self):
        # Check for a nontrival function that spectral density (e.g. x**2) is correctly computed
        params_comp = self.params_KernelMatrix.copy()
        params_comp["spec_dens"] = lambda x: np.exp(-x**2)

        K_comp = KernelMatrix(**params_comp)
        spec_dens_array = K_comp.spec_dens_array_cmplx
        fine_grid_complex = K_comp.fine_grid * np.exp(1j * K_comp.phi)
        spec_dens_array_check = np.array([np.exp(-x**2) for x in fine_grid_complex])
        self.assertTrue(np.array_equal(spec_dens_array, spec_dens_array_check))


if __name__ == "__main__":
    unittest.main()
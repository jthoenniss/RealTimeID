import unittest
import numpy as np
import src.utils.common_funcs as cf
from src.dlr_kernel.dlr_kernel import (
    DecompKernel,
    DlrKernel,
)  # Adjust the import according to your project structure


class TestRtKernel(unittest.TestCase):
    def setUp(self):
        # Common setup that can be used across multiple tests

        self.K = DecompKernel(
            m=10,
            n=5,
            beta=1.0,
            N_max = 10,
            delta_t = 0.1,
            eps=0.1,
            h=0.2,
            phi=np.pi / 4,
        )

    def test_initialization(self):
        # Test initialization of RtKernel
        self.assertEqual(self.K.m, 10)
        self.assertEqual(self.K.n, 5)
        self.assertEqual(self.K.beta, 1.0)
        self.assertEqual(self.K.N_max, 10)
        self.assertEqual(self.K.delta_t, 0.1)
        self.assertTrue(np.array_equal(self.K.times, cf.set_time_grid(N_max=10, delta_t=0.1)))
        self.assertEqual(self.K.eps, 0.1)
        self.assertEqual(self.K.h, 0.2)
        self.assertEqual(self.K.phi, np.pi / 4)

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

    def test_perform_svd_maxrank(self):
        # Test SVD for corner case of error eps = 0.0 
        num_sv, sv = self.K.perform_svd(eps = 0.0)
        expected_num_sv = min(self.K.kernel.shape[0], self.K.kernel.shape[1])
        self.assertEqual(num_sv, expected_num_sv)

    def test_perform_svd_minrank(self):
        # Test SVD for corner case of error eps = 1.0
        num_sv, sv = self.K.perform_svd(eps = 1.0)
        expected_num_sv = 0
        self.assertEqual(num_sv, expected_num_sv)
        
    def test_perform_ID_maxrank(self):
        # Test SID for corner case of error eps = 0.0 
        ID_rank, idx, proj = self.K.perform_ID(eps = 0.0)
        expected_ID_rank = min(self.K.kernel.shape[0], self.K.kernel.shape[1])
        self.assertEqual(ID_rank, expected_ID_rank)


class TestRtDlr_with_kwargs(unittest.TestCase):
    def setUp(self):
        # Common setup for RtDlr tests
        self.dlr = DlrKernel(
            m=10,
            n=5,
            beta=1.0,
            N_max = 10,
            delta_t = 0.1,
            eps=0.1,
            h=0.2,
            phi=np.pi / 4,
        )

    def test_initialization(self):
        # Test initialization of RtDlr
        self.assertEqual(self.dlr.m, 10)
        self.assertEqual(self.dlr.n, 5)
        self.assertEqual(self.dlr.beta, 1.0)
        self.assertEqual(self.dlr.delta_t, 0.1)
        self.assertTrue(np.array_equal(self.dlr.times, cf.set_time_grid(N_max=10, delta_t=0.1)))
        self.assertEqual(self.dlr.eps, 0.1)
        self.assertEqual(self.dlr.h, 0.2)
        self.assertEqual(self.dlr.phi, np.pi / 4)

    def test_spec_dens_fine(self):
        # Test spectral density calculation
        spec_dens = self.dlr.spec_dens_fine()
        # Add assertions to validate spec_dens

    def test_get_projection_matrix(self):
        # Test projection matrix calculation
        P = self.dlr.get_projection_matrix()
        # Add assertions to validate P

    def test_reconstruct_propag(self):
        # Test propagator reconstruction
        G_reconstr = next(self.dlr.reconstruct_propag(compute_error=False))
        # Add assertions to validate G_reconstr

        # Optionally test error computation
        G_reconstr, error_rel = self.dlr.reconstruct_propag(compute_error=True)
        # Add assertions to validate error_rel

    # Add more test methods as necessary...


if __name__ == "__main__":
    unittest.main()

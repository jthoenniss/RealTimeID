import unittest
import numpy as np
from src.rt_kernel.rt_kernel import (
    RtKernel,
    RtDlr,
)  # Adjust the import according to your project structure


class TestRtKernel(unittest.TestCase):
    def setUp(self):
        # Common setup that can be used across multiple tests

        self.kernel = RtKernel(
            m=10,
            n=5,
            beta=1.0,
            times=np.array([0.0, 1.0, 2.0]),
            eps=0.1,
            h=0.2,
            phi=np.pi / 4,
        )

    def test_initialization(self):
        # Test initialization of RtKernel
        self.assertEqual(self.kernel.m, 10)
        self.assertEqual(self.kernel.n, 5)
        self.assertEqual(self.kernel.beta, 1.0)
        self.assertTrue(np.array_equal(self.kernel.times, np.array([0.0, 1.0, 2.0])))
        self.assertEqual(self.kernel.eps, 0.1)
        self.assertEqual(self.kernel.h, 0.2)
        self.assertEqual(self.kernel.phi, np.pi / 4)

    def test_create_kernel(self):
        # Test the kernel creation method
        fine_grid, K = self.kernel.create_kernel()
        k_vals = np.arange(-self.kernel.n, self.kernel.m + 1)
        fine_grid_check = np.exp(
            self.kernel.h * k_vals - np.exp(-self.kernel.h * k_vals)
        )
        self.assertTrue(np.array_equal(fine_grid, fine_grid_check))

    def test_kernel_matrix_shape(self):
        _, K = self.kernel.create_kernel()
        expected_shape = (len(self.kernel.times), self.kernel.m + self.kernel.n + 1)
        self.assertEqual(K.shape, expected_shape)

    def test_perform_svd_maxrank(self):
        # Test SVD for corner case of error eps = 0.0 
        num_sv, sv = self.kernel.perform_svd(eps = 0.0)
        expected_num_sv = min(self.kernel.K.shape[0], self.kernel.K.shape[1])
        self.assertEqual(num_sv, expected_num_sv)

    def test_perform_svd_minrank(self):
        # Test SVD for corner case of error eps = 1.0
        num_sv, sv = self.kernel.perform_svd(eps = 1.0)
        expected_num_sv = 0
        self.assertEqual(num_sv, expected_num_sv)
        
    def test_perform_ID_maxrank(self):
        # Test SID for corner case of error eps = 0.0 
        ID_rank, idx, proj = self.kernel.perform_ID(eps = 0.0)
        expected_ID_rank = min(self.kernel.K.shape[0], self.kernel.K.shape[1])
        self.assertEqual(ID_rank, expected_ID_rank)


class TestRtDlr_with_kwargs(unittest.TestCase):
    def setUp(self):
        # Common setup for RtDlr tests
        self.dlr = RtDlr(
            m=10,
            n=5,
            beta=1.0,
            times=np.array([0.0, 1.0, 2.0]),
            eps=0.1,
            h=0.2,
            phi=np.pi / 4,
        )

    def test_initialization(self):
        # Test initialization of RtDlr
        self.assertEqual(self.dlr.m, 10)
        self.assertEqual(self.dlr.n, 5)
        self.assertEqual(self.dlr.beta, 1.0)
        self.assertTrue(np.array_equal(self.dlr.times, np.array([0.0, 1.0, 2.0])))
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

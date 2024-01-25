import unittest
import numpy as np
import src.utils.common_funcs as cf
from src.decomp_kernel.decomp_kernel import DecompKernel
from src.discr_error.discr_error import DiscrError
from src.spec_dens.spec_dens import spec_dens_gapless

class TestDecompKernel_with_kwargs(unittest.TestCase):
    
    def setUp(self):
        # Common setup that can be used across multiple tests

        self.K = DecompKernel(
            m=10,
            n=5,
            beta=1.0,
            N_max=10,
            delta_t=0.1,
            eps=0.1,
            h=0.2,
            phi=np.pi / 4,
            spec_dens = lambda x: spec_dens_gapless(x)
        )

    def test_initialization(self):
        # Test initialization of RtKernel
        self.assertEqual(self.K.m, 10)
        self.assertEqual(self.K.n, 5)
        self.assertEqual(self.K.beta, 1.0)
        self.assertEqual(self.K.N_max, 10)
        self.assertEqual(self.K.delta_t, 0.1)
        self.assertTrue(
            np.array_equal(self.K.times, cf.set_time_grid(N_max=10, delta_t=0.1))
        )
        self.assertEqual(self.K.eps, 0.1)
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

    def test_perform_SVD_maxrank(self):
        # Test SVD for corner case of error eps = 0.0
        num_sv, sv = self.K.perform_SVD(eps=0.0)
        expected_num_sv = min(self.K.kernel.shape[0], self.K.kernel.shape[1])
        self.assertEqual(num_sv, expected_num_sv)

    def test_perform_SVD_minrank(self):
        # Test SVD for corner case of error eps = 1.0
        num_sv, sv = self.K.perform_SVD(eps=1.0)
        expected_num_sv = 0
        self.assertEqual(num_sv, expected_num_sv)

    def test_perform_ID_maxrank(self):
        # Test SID for corner case of error eps = 0.0
        ID_rank, idx, proj = self.K.perform_ID(eps=0.0)
        expected_ID_rank = min(self.K.kernel.shape[0], self.K.kernel.shape[1])
        self.assertEqual(ID_rank, expected_ID_rank)

    def test_get_vars(self):
        #set if all required keys are present in DecompKernel object
        REQUIRED_KEYS = {
            "m",
            "n",
            "beta",
            "N_max",
            "delta_t",
            "h",
            "phi",
            "ID_rank",
            "proj",
            "idx",
            "fine_grid",
            "coarse_grid",
            "singular_values",
            "nbr_sv_above_eps",
            "spec_dens",
            "spec_dens_array_fine"
        }

        # dictionary of all class attributes
        attrs = vars(self.K)

        # check that all required keys are present in kernel object
        for key in REQUIRED_KEYS:
            self.assertTrue(
                key in attrs.keys(), f"Key '{key}' not found in DecompKernel"
            )


class TestDecompKernel_with_DiscError(unittest.TestCase):
    """
    This test class checks if initialization is equivalent when using kwargs and when using an instance of DiscrError
    """

    def setUp(self):
        params_DiscrError = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
            "upper_cutoff": 600,
            "spec_dens": lambda x: spec_dens_gapless(x)
        }
        D = DiscrError(**params_DiscrError)

        self.dck = DecompKernel(D)

        # Add eps to params, such that both initilizations should be equivalent,
        # Also: pop upper_cutoff
        params_DiscrError["eps"] = D.eps
        params_DiscrError.pop("upper_cutoff")

        self.dck_kwargs = DecompKernel(**params_DiscrError)

    def test_base_attrs_present(self):
        KernelMatrix_keys_required = [
            "m",
            "n",
            "beta",
            "N_max",
            "delta_t",
            "h",
            "phi",
            "times",
            "fine_grid",
            "k_values",
            "kernel",
            "spec_dens",
            "spec_dens_array_fine"
        ]

        present_keys = vars(self.dck).keys()
        for key in KernelMatrix_keys_required:  # check that all attributes exist.
            self.assertTrue(key in present_keys, f"Key {key} not found.")

    def test_initialization(self):
        # check that initialization is equivalent to initialization with kwargs
        for key, val in vars(self.dck).items():
      
            if key != "spec_dens":
                self.assertTrue(
                    np.allclose(val,
                    getattr(self.dck_kwargs, key)),
                    f"The following attributes differs: {key},{val}, {getattr(self.dck_kwargs, key)}",
                )
            else:
                self.assertEqual(val, getattr(self.dck_kwargs, key))
                self.assertEqual(val(0), 1.)


if __name__ == "__main__":
    unittest.main()

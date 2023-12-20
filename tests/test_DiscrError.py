import unittest
import numpy as np
import src.utils.common_funcs as cf
from src.discr_error.discr_error import DiscrError


class TestDiscError(unittest.TestCase):
    def setUp(self):

        params_DiscrError = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
            "upper_cutoff" : 600
        }
        self.D = DiscrError(**params_DiscrError)



    def test_base_attrs_present(self):
        KernelMatrix_keys_required = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel"]

        KernelMatrix_keys = vars(self.D).keys()

        for key in KernelMatrix_keys_required:#check that all attributes exist.
            self.assertTrue(key in KernelMatrix_keys, f"Key {key} not found.")

    def test_initialization(self):
        
        # Test initialization of RtDlr
        self.assertEqual(self.D.m, 10)
        self.assertEqual(self.D.n, 5)
        self.assertEqual(self.D.beta, 1.0)
        self.assertEqual(self.D.delta_t, 0.1)
        self.assertTrue(
            np.array_equal(self.D.times, cf.set_time_grid(N_max=10, delta_t=0.1))
        )
        self.assertEqual(self.D.h, 0.2)
        self.assertEqual(self.D.phi, np.pi / 4)

    def test_eps(self):#check that relative error is leq 1.
        self.assertLessEqual(self.D.eps, 1.0)

if __name__ == "__main__":
    unittest.main()
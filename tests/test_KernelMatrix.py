import unittest
import numpy as np
from src.kernel_matrix.kernel_matrix import KernelMatrix


class TestKernelMatrix(unittest.TestCase):
    def setUp(self):

        params_KernelMatrix = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "h": 0.2,
            "phi": np.pi / 4
        }
        self.K = KernelMatrix(**params_KernelMatrix)


    def test_all_attrs_present(self):
        KernelMatrix_keys_required = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel"]

        KernelMatrix_keys = vars(self.K).keys()

        for key in KernelMatrix_keys_required:#check that all attributes exist.
            self.assertTrue(key in KernelMatrix_keys, f"Key {key} not found.")

if __name__ == "__main__":
    unittest.main()
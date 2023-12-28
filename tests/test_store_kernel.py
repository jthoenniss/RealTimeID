import unittest
import numpy as np
from src.decomp_kernel.decomp_kernel import DecompKernel
from src.store_kernel.store_kernel import store_kernel

class Test_store_kernel(unittest.TestCase):
    def setUp(self):
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
        }

        params_DecompKernel = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "eps": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
        }

        self.kernel = DecompKernel(**params_DecompKernel)

    def test_store(self):

        store_kernel(self)

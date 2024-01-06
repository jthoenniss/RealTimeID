import unittest
import numpy as np
import src.utils.common_funcs as cf
from src.dlr_kernel.dlr_kernel import DlrKernel
from src.discr_error.discr_error import DiscrError


class TestDlrKernel_with_kwargs(unittest.TestCase):
    def setUp(self):
        params_DlrKernel = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "eps": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
            "spec_dens": lambda x: 1.
        }
        self.dlr = DlrKernel(**params_DlrKernel)

    def test_initialization(self):
        # Test initialization of RtDlr
        self.assertEqual(self.dlr.m, 10)
        self.assertEqual(self.dlr.n, 5)
        self.assertEqual(self.dlr.beta, 1.0)
        self.assertEqual(self.dlr.delta_t, 0.1)
        self.assertTrue(
            np.array_equal(self.dlr.times, cf.set_time_grid(N_max=10, delta_t=0.1))
        )
        self.assertEqual(self.dlr.eps, 0.1)
        self.assertEqual(self.dlr.h, 0.2)
        self.assertEqual(self.dlr.phi, np.pi / 4)
        self.assertEqual(self.dlr.spec_dens(0), 1.)


class TestDlrKernel_with_DiscError(unittest.TestCase):
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
            "upper_cutoff" : 600,
            "spec_dens": lambda x: np.ones_like(np.atleast_1d(x))
        }
        D = DiscrError(**params_DiscrError)

        self.dlr = DlrKernel(D)

        #Add eps to params, such that both initilizations should be equivalent,
        #Also: pop upper_cutoff
        params_DiscrError["eps"] = D.eps
        params_DiscrError.pop("upper_cutoff")

        self.dlr_kwargs = DlrKernel(**params_DiscrError)

    def test_base_attrs_present(self):
        KernelMatrix_keys_required = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel", "spec_dens", "spec_dens_array_cmplx"]

        present_keys = vars(self.dlr).keys()
        for key in KernelMatrix_keys_required:#check that all attributes exist.
            self.assertTrue(key in present_keys, f"Key {key} not found.")


    def test_initialization(self):
        # check that initialization is equivalent to initialization with kwargs
        for key, val in vars(self.dlr).items():
        
            if key != "spec_dens":
                        self.assertTrue(
                            np.allclose(val,
                            getattr(self.dlr_kwargs, key)),
                            f"The following attributes differs: {key},{val}, {getattr(self.dlr_kwargs, key)}",
                        )
            else:
                self.assertEqual(val, getattr(self.dlr_kwargs, key))
                self.assertEqual(val(0), 1.)

if __name__ == "__main__":
    unittest.main()

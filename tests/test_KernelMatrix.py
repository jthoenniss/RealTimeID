import unittest
import numpy as np
import os, sys
project_path = os.environ.get('REALTIMEID_PATH')
if project_path and project_path not in sys.path:
    sys.path.append(project_path)
    print("Project path successfully added.")


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
            "spec_dens":  lambda x: spec_dens_gapless(x),
            "freq_parametrization": "fancy_exp",
        }
        self.K = KernelMatrix(**self.params_KernelMatrix)


    def test_all_attrs_present(self):
        KernelMatrix_keys_required = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel", "spec_dens", "freq_parametrization"]

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
        self.assertEqual(self.K.freq_parametrization, "fancy_exp")

    def test_create_kernel(self):
        # Test the kernel creation method
        fine_grid, k_values = self.K.fine_grid, self.K.k_values
        k_values_check = np.arange(-self.K.n, self.K.m + 1)
        #check that the fine grid is correctly computed for the fancy_exp parametrization
        fine_grid_check = np.exp(
            self.K.h * k_values_check - np.exp(-self.K.h * k_values_check)
        )
        self.assertTrue(np.array_equal(k_values, k_values_check))
        self.assertTrue(np.array_equal(fine_grid, fine_grid_check))

        #test values of kernel matrix:
        times = cf.set_time_grid(N_max=self.params_KernelMatrix["N_max"], delta_t=self.params_KernelMatrix["delta_t"])
        times_arr = times[:, np.newaxis]  # enable broadcasting
        fine_grid_complex = fine_grid_check * np.exp(1j * self.K.phi)
        #add negative frequencies
        fine_grid_complex = np.concatenate((-fine_grid_complex[::-1].conj(), fine_grid_complex))
        K_check_particle = np.exp(1.j * times_arr * fine_grid_complex) / (1 + np.exp(- self.K.beta * fine_grid_complex)) * self.params_KernelMatrix["spec_dens"](fine_grid_complex)
        K_check_hole = np.exp(1.j *  times_arr * fine_grid_complex) / (1 + np.exp( self.K.beta * fine_grid_complex)) * self.params_KernelMatrix["spec_dens"](fine_grid_complex)

        #combine particle and hole contributions
        K_check = np.vstack((K_check_particle, K_check_hole))

        jacobian = self.K.h * (1 + np.exp(-self.K.h * k_values_check)) * fine_grid * np.exp(1.j * self.params_KernelMatrix["phi"])
        #add negative frequencies to Jacobian
        jacobian_complex = np.concatenate((jacobian[::-1], jacobian))

        K_check *= jacobian_complex # multiply by Jacobian from measure. This is dw/dk. 

        self.assertTrue(np.allclose(self.K.kernel, K_check))

    def test_kernel_matrix_shape(self):
        K = self.K.kernel
        expected_shape = (2*len(self.K.times), 2*(self.K.m + self.K.n + 1)) # first factor 2: particle and holes substacked, second factor 2: negative and positive frequencies
        self.assertEqual(K.shape, expected_shape)



    def test_simple_exp_freq_parametrization(self):
        #create new KernelMatrix object, this time with simple-exponential frequency parametrization, i.e. freq_parametrization = "simple_exp"
        params_comp = self.params_KernelMatrix.copy()
        params_comp["freq_parametrization"] = "simple_exp" # if this keyword is removed, the grid should default to the simple exponential grid
        K_simple_exp = KernelMatrix(**params_comp)

        fine_grid, k_values = K_simple_exp.fine_grid, K_simple_exp.k_values
        k_values_check = np.arange(-K_simple_exp.n, K_simple_exp.m + 1)
        #check that the fine grid is correctly computed for the fancy_exp parametrization
        fine_grid_check = np.exp(K_simple_exp.h * k_values_check)

        self.assertTrue(np.array_equal(k_values, k_values_check))
        self.assertTrue(np.array_equal(fine_grid, fine_grid_check))

        #test values of kernel matrix:
        times = cf.set_time_grid(N_max=self.params_KernelMatrix["N_max"], delta_t=self.params_KernelMatrix["delta_t"])
        times_arr = times[:, np.newaxis]  # enable broadcasting
        fine_grid_complex = fine_grid_check * np.exp(1j * K_simple_exp.phi)
        #add negative frequencies
        fine_grid_complex = np.concatenate((-fine_grid_complex[::-1].conj(), fine_grid_complex))
        K_check_particle = cf.dynamic_distr_particle(times_arr, fine_grid_complex, self.K.beta) * self.params_KernelMatrix["spec_dens"](fine_grid_complex)
        K_check_hole = cf.dynamic_distr_particle(times_arr, fine_grid_complex, -self.K.beta)  * self.params_KernelMatrix["spec_dens"](fine_grid_complex)

        #combine particle and hole contributions
        K_check = np.vstack((K_check_particle, K_check_hole))

        jacobian = self.K.h * fine_grid * np.exp(1.j * self.params_KernelMatrix["phi"])
        #add negative frequencies to Jacobian
        jacobian_complex = np.concatenate((jacobian[::-1], jacobian))

        K_check *= jacobian_complex # multiply by Jacobian from measure. This is dw/dk. Multiply by exp(i*phi) to rotate in the complex plane

        self.assertTrue(np.allclose(K_simple_exp.kernel, K_check))

if __name__ == "__main__":
    unittest.main()
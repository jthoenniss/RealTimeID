import unittest
import numpy as np
import src.utils.common_funcs as cf
from src.discr_error.discr_error import DiscrError
from src.kernel_params.kernel_params import KernelParams
import math #for floor and ceiling

class TestDiscError(unittest.TestCase):
    def setUp(self):

        self.params_DiscrError = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
            "upper_cutoff" : 600,
            "spec_dens": lambda x: 1.,
        }
        self.D = DiscrError(**self.params_DiscrError)



    def test_base_attrs_present(self):
        KernelMatrix_keys_required = ["m","n","beta","N_max","delta_t", "h","phi","times","fine_grid","k_values","kernel", "spec_dens", "spec_dens_array_cmplx"]

        KernelMatrix_keys = vars(self.D).keys()

        for key in KernelMatrix_keys_required:#check that all attributes exist.
            self.assertTrue(key in KernelMatrix_keys, f"Key {key} not found.")

    def test_initialization(self):
        
        # Test initialization of DiscrError
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


    def test_get_reduced_kernel_and_error(self):
        # Test that reduced kernel and error are computed correctly
        upper_idx = len(self.D.fine_grid) - 4
        lower_idx = 2
        grids_reduced = self.D._get_reduced_kernel_and_error(lower_idx=lower_idx, upper_idx=upper_idx)

        time_grid = cf.set_time_grid(N_max=self.params_DiscrError["N_max"], delta_t=self.params_DiscrError["delta_t"])
        nbr_times = len(time_grid)

        #check shapes
        self.assertEqual(grids_reduced["kernel_reduced"].shape, (nbr_times, upper_idx - lower_idx))
        self.assertEqual(grids_reduced["spec_dens_array_cmplx_reduced"].shape, (upper_idx - lower_idx,))
        self.assertEqual(grids_reduced["discrete_integral_reduced"].shape, (nbr_times,))
        self.assertEqual(grids_reduced["eps_reduced"].shape, ())

        #check that correct sub arrays are returned
        self.assertTrue(np.array_equal(grids_reduced["kernel_reduced"], self.D.kernel[:, lower_idx:upper_idx]))
        self.assertTrue(np.array_equal(grids_reduced["spec_dens_array_cmplx_reduced"], self.D.spec_dens_array_cmplx[lower_idx:upper_idx]))
        
        
        #______Check that the discrete integral and error are computed correctly_____
        #compute new instance of DiscrError with reduced kernel and error
        params_comp = self.params_DiscrError.copy()
        params_comp["m"] =  6
        params_comp["n"] = 3
        D_comp = DiscrError(**params_comp)
        self.assertTrue(np.allclose(grids_reduced["discrete_integral_reduced"], D_comp.discrete_integral_init[:]))

        #grids_reduced["eps_reduced"] is the relative error between the reduced and the orginial discretized 
        eps_comp = D_comp.error_time_integrated(time_series_exact = self.D.discrete_integral_init)
        self.assertTrue(np.allclose(grids_reduced["eps_reduced"],eps_comp))


    def test_optimize_mode_count(self):
        nbr_freqs = len(self.D.fine_grid)
        
        #_______CASE 1: rel_error_diff = 0_______
        #if threshold for additional error from shrinked grid is 0, 
        #then the mode count should yield 0.
        rel_error_diff = 0
        m_count = self.D._optimize_mode_count(self.params_DiscrError["m"], lambda mc: [0, nbr_freqs - mc], rel_error_diff=rel_error_diff)
        n_count = self.D._optimize_mode_count(self.params_DiscrError["m"], lambda nc: [nc, nbr_freqs], rel_error_diff=rel_error_diff)

        self.assertEqual(m_count, 0)
        self.assertEqual(n_count, 0)


        #_______CASE 2: rel_error_diff = np.inf_______
        #if threshold for additional error from shrinked grid is np.inf, 
        #then the mode count should yield the maximum count m-1 or n-1, resp.
        rel_error_diff = np.inf
        m_count = self.D._optimize_mode_count(self.params_DiscrError["m"], lambda mc: [0, nbr_freqs - mc], rel_error_diff=rel_error_diff)
        n_count = self.D._optimize_mode_count(self.params_DiscrError["n"], lambda nc: [nc, nbr_freqs], rel_error_diff=rel_error_diff)
        
        self.assertEqual(m_count, self.params_DiscrError["m"]-1)
        self.assertEqual(n_count, self.params_DiscrError["n"]-1)


    def test_optimize_rel_error_0(self):

        #_______CASE: rel_error_diff = 0_______
        #if threshold for additional error from shrinked grid is 0, 
        #then 'm' and 'n' should not be changed during optimization.
        params = KernelParams(**self.params_DiscrError)
        rel_error_diff = 0
        D_opt = self.D.optimize(update_params=params, rel_error_diff=rel_error_diff)
        self.assertEqual(D_opt.m, self.params_DiscrError["m"])
        self.assertEqual(D_opt.n, self.params_DiscrError["n"])

        #check that external parameters are updated correctly
        self.assertEqual(D_opt.m, params.get_param("m"))
        self.assertEqual(D_opt.n, params.get_param("n"))
        discrete_cutoffs = params.get_discrete_cutoffs()
        lowest_freq, largest_freq = np.exp(-discrete_cutoffs[0] - np.exp(discrete_cutoffs[0])), np.exp(discrete_cutoffs[1] - np.exp(-discrete_cutoffs[1]))
        self.assertEqual(D_opt.fine_grid[0], lowest_freq)#lower discrete cutoff
        self.assertEqual(D_opt.fine_grid[-1], largest_freq)#upper discrete cutoff

    def test_optimize_rel_error_inf(self):

        #_______CASE: rel_error_diff = np.inf_______
        #if threshold for additional error from shrinked grid is np.inf, 
        #then 'm' and 'n' should be reduced by 1 during optimization.
        params = KernelParams(**self.params_DiscrError)
        rel_error_diff = np.inf
        D_opt = self.D.optimize(update_params=params, rel_error_diff=rel_error_diff)
        self.assertEqual(D_opt.m, 1)
        self.assertEqual(D_opt.n, 1)
        
        #check that external parameters are updated correctly
        self.assertEqual(D_opt.m, params.get_param("m"))
        self.assertEqual(D_opt.n, params.get_param("n"))
        discrete_cutoffs = params.get_discrete_cutoffs()
        lowest_freq, largest_freq = np.exp(-discrete_cutoffs[0] - np.exp(discrete_cutoffs[0])), np.exp(discrete_cutoffs[1] - np.exp(-discrete_cutoffs[1]))
        self.assertEqual(D_opt.fine_grid[0], lowest_freq)#lower discrete cutoff
        self.assertEqual(D_opt.fine_grid[-1], largest_freq)#upper discrete cutoff

if __name__ == "__main__":
    unittest.main()
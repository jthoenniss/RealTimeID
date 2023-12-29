import unittest
import numpy as np
from src.decomp_kernel.decomp_kernel import DecompKernel
from src.store_kernel.store_kernel import Hdf5Kernel
from src.utils import common_funcs as cf
import os

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

        self.params_DecompKernel = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "eps": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
        }

        self.kernel = DecompKernel(**self.params_DecompKernel)

        name_template = 'data_temp/beta={}_delta_t={}_unittest.h5'
        self._filename = name_template.format(self.params_DecompKernel["beta"], self.params_DecompKernel["delta_t"])
        


    def test_store_single_element(self):

        #store a single kernel object
        hdf_kernel = Hdf5Kernel(filename=self._filename)
        hdf_kernel.store_kernel_data(self.kernel)
    
        #read it back out and see if correct values are obtained
        params, data  = hdf_kernel.read_kernel_element(0, isFlatIndex=True)
        
        self.assertTrue(params, self.params_DecompKernel)


    
    def test_store_array(self):

        #store an array of kernel objects
        hdf_kernel = Hdf5Kernel(filename=self._filename)
        array = np.array([[self.kernel,self.kernel],[self.kernel,self.kernel]])
        hdf_kernel.store_kernel_data(array)
    
        #read out a single element and see if correct values are obtained
        params, data  = hdf_kernel.read_kernel_element((0,0))
        self.assertTrue(params, self.params_DecompKernel)
        self.assertTrue(data["ID_rank"], self.kernel.ID_rank)
        params, data  = hdf_kernel.read_kernel_element((0,1))
        self.assertTrue(params, self.params_DecompKernel)
        self.assertTrue(data["ID_rank"], self.kernel.ID_rank)
        params, data  = hdf_kernel.read_kernel_element((1,0))
        self.assertTrue(params, self.params_DecompKernel)
        self.assertTrue(data["ID_rank"], self.kernel.ID_rank)
        params, data  = hdf_kernel.read_kernel_element((1,1))
        self.assertTrue(params, self.params_DecompKernel)
        self.assertTrue(data["ID_rank"], self.kernel.ID_rank)



    def test_read_to_array(self):

        #store an array of kernel objects
        hdf_kernel = Hdf5Kernel(filename=self._filename)
        array = np.array([[self.kernel,self.kernel],[self.kernel,self.kernel]])
        hdf_kernel.store_kernel_data(array)
    
        #read out a single element and see if correct values are obtained
        
        errors, m_vals, n_vals, h_vals, ID_ranks  = hdf_kernel.read_to_array()

        #check that data is equal to data directly extracted from kernel objects
        errors_comp, m_vals_comp, n_vals_comp, h_vals_comp, ID_ranks_comp = cf.create_numpy_arrays_from_kernel(array)
        self.assertTrue(np.allclose(errors, errors_comp))
        self.assertTrue(np.allclose(m_vals, m_vals_comp))
        self.assertTrue(np.allclose(n_vals, n_vals_comp))
        self.assertTrue(np.allclose(h_vals, h_vals_comp))
        self.assertTrue(np.allclose(ID_ranks, ID_ranks_comp))


        #check that data is equal to data taken from a single kernel (all kernels in the array are equivalent here)
        self.assertTrue(np.all([val == self.kernel.eps for val in errors]))
        self.assertTrue(np.all([val == self.kernel.m for val in m_vals]))
        self.assertTrue(np.all([val == self.kernel.n for val in n_vals]))
        self.assertTrue(np.all([val == self.kernel.h for val in h_vals]))
        self.assertTrue(np.all([val == self.kernel.ID_rank for val in ID_ranks]))

    def tearDown(self):
        # This method runs after each test and cleans up temporarily created hdf5 files that are no longer needed
        if os.path.exists(self._filename):
            os.remove(self._filename)
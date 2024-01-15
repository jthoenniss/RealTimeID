import unittest
import numpy as np
from src.decomp_kernel.decomp_kernel import DecompKernel
from src.discr_error.discr_error import DiscrError
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
            "spec_dens_array_cmplx"
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
            "spec_dens": lambda x: 1.
        }

        self.kernel = DecompKernel(**self.params_DecompKernel)
        

        name_template = 'data_temp/beta={}_delta_t={}_unittest.h5'
        self._filename = name_template.format(self.params_DecompKernel["beta"], self.params_DecompKernel["delta_t"])
        


    def test_store_single_element(self):

        #store a single kernel object
        hdf_kernel = Hdf5Kernel(filename=self._filename).create_file(kernel_dims=(1,))


        hdf_kernel.store_kernel_array(np.array([self.kernel]))
    
        #read it back out and see if correct values are obtained
        params, data  = hdf_kernel.read_kernel_element(0, isFlatIndex=True)#0 is the flat index in the parameter point array.
        
        self.assertTrue(params, self.params_DecompKernel)


    
    def test_store_array(self):
        #ToDo: Extend to test all other attributes

        #store an array of kernel objects directly by feeding array
        arr = np.array([[self.kernel,self.kernel],[self.kernel,self.kernel]])
        
        #create file
        hdf_kernel = Hdf5Kernel(filename=self._filename).create_file(kernel_dims=arr.shape)
        #store data
        hdf_kernel.store_kernel_array(arr)
    
        #read out single elements and see if correct values are obtained
        for i in range (2):
            for j in range (2):
                params, data  = hdf_kernel.read_kernel_element((i,j))
                
                for key, val in params.items():
                    if key != "spec_dens_array_cmplx":
                        self.assertTrue(np.allclose(val,self.params_DecompKernel[key]), f"parameters differ for key {key}")

                    else:
                        self.assertEqual(val, getattr(self.params_DecompKernel, key), f"parameters differ for key {key}")
                        self.assertEqual(val(0), 1.)
                        
                self.assertEqual(data["ID_rank"], self.kernel.ID_rank)
                self.assertTrue(np.allclose(data["spec_dens_array_cmplx"], self.kernel.spec_dens_array_cmplx))


    def test_append_element(self):
        #ToDo: Extend to test all other attributes

        #store an array of kernel objects directly by feeding array
        arr = np.array([[self.kernel,self.kernel],[self.kernel,self.kernel]])
        
        #create file
        hdf_kernel = Hdf5Kernel(filename=self._filename).create_file(kernel_dims=arr.shape)

        for i in range (2):
            for j in range (2):
                #append data
                hdf_kernel.append_kernel_element(idx = (i,j), kernel_object=arr[i,j])
    
    
        #read out single elements and see if correct values are obtained
        for i in range (2):
            for j in range (2):
                params, data  = hdf_kernel.read_kernel_element((i,j))
     
                for key, val in params.items():
                    if key != "spec_dens_array_cmplx":
                        self.assertTrue(np.allclose(val,self.params_DecompKernel[key]), f"parameters differ for key {key}")

                    else:
                        self.assertEqual(val, self.params_DecompKernel[key], f"parameters differ for key {key}")
                        self.assertEqual(val(0), 1.)
                        

                self.assertEqual(data["ID_rank"], self.kernel.ID_rank)
                self.assertTrue(np.allclose(data["spec_dens_array_cmplx"], self.kernel.spec_dens_array_cmplx))

    def test_append_element_two_kernels_and_dict(self):
        #take over parameters specified in setUp
        params_DiscrError = self.params_DecompKernel.copy()
        #eps not needed as parameter to initialize, remove it
        del params_DiscrError["eps"]

        #first, create DiscrError kernel 
        kernel_discr = DiscrError(**params_DiscrError, upper_cutoff= 10)
        #then create DecompKernel from DiscrError instance
        kernel_decomp = DecompKernel(kernel_discr)

        test_data = np.arange(10)
        dict_supp = {"test_data": test_data}
        
        #store an array of kernel objects directly by feeding array
        arr_decomp = np.array([[kernel_decomp,kernel_decomp],[kernel_decomp,kernel_decomp]])
        arr_discr = np.array([[kernel_discr,kernel_discr],[kernel_discr,kernel_discr]])
        arr_dict = np.array([[dict_supp,dict_supp],[dict_supp,dict_supp ]])

        #create file
        hdf_kernel = Hdf5Kernel(filename=self._filename).create_file(kernel_dims=arr_decomp.shape)

        for i in range (2):
            for j in range (2):
                #append data: Store both kernels and a dictionary
                hdf_kernel.append_kernel_element(idx = (i,j), kernel_object=arr_decomp[i,j], kernel_object2=arr_discr[i,j], dict_data=arr_dict[i,j])
    
    
        #read out single elements and see if correct values are obtained
        for i in range (2):
            for j in range (2):
                params, data  = hdf_kernel.read_kernel_element((i,j))
     
                #check that parameters are equal
                for key, val in params.items():  
                    self.assertTrue(np.allclose(val,getattr(kernel_discr,key)), f"parameters differ for key {key}: {val, self.params_DecompKernel[key]}")

                #check that data is equal to data directly extracted from kernel objects
                kernel_decomp_keys = vars(kernel_decomp).keys()
                kernel_discr_keys = vars(kernel_discr).keys()

                for key, val in data.items():  
                    if key in kernel_decomp_keys:
                        self.assertTrue(np.allclose(val, getattr(kernel_decomp, key)), f"Data differs for key {key}")
                    elif key in kernel_discr_keys:
                        self.assertTrue(np.allclose(val, getattr(kernel_discr, key)), f"Data differs for key {key}")
                    elif key in dict_supp.keys():
                        self.assertTrue(np.allclose(val, dict_supp.get(key)), f"Data differs for key {key}")
                    else:
                        raise ValueError(f"Key {key} not found in kernel objects or dictionary.")

                    


    def test_read_to_array(self):
        #ToDo: Extend to test all other attributes

        #store an array of kernel objects
        hdf_kernel = Hdf5Kernel(filename=self._filename).create_file(kernel_dims=(2,2))
        array = np.array([[self.kernel,self.kernel],[self.kernel,self.kernel]])
        hdf_kernel.store_kernel_array(array)
    
        #read out a single element and see if correct values are obtained
        
        data, kernel_dims  = hdf_kernel.read_to_array()

        errors = data["errors"]
        m_vals = data["m_vals"]
        n_vals = data["n_vals"]
        h_vals = data["h_vals"]
        ID_ranks = data["ID_ranks"]
        N_maxs = data["N_maxs"]
        betas = data["betas"]
        delta_t_vals = data["delta_t_vals"]

        #check that data is equal to data directly extracted from kernel objects
        data_comp, kernel_dims_comp = cf.create_numpy_arrays_from_kernel(array)

        errors_comp = data_comp["errors"]
        m_vals_comp = data_comp["m_vals"]
        n_vals_comp = data_comp["n_vals"]
        h_vals_comp = data_comp["h_vals"]
        ID_ranks_comp = data_comp["ID_ranks"]
        N_maxs_comp = data_comp["N_maxs"]
        betas_comp = data_comp["betas"]
        delta_t_vals_comp = data_comp["delta_t_vals"]

        self.assertTrue(np.allclose(errors, errors_comp))
        self.assertTrue(np.allclose(m_vals, m_vals_comp))
        self.assertTrue(np.allclose(n_vals, n_vals_comp))
        self.assertTrue(np.allclose(h_vals, h_vals_comp))
        self.assertTrue(np.allclose(ID_ranks, ID_ranks_comp))
        self.assertTrue(np.allclose(N_maxs, N_maxs_comp))
        self.assertTrue(np.allclose(betas, betas_comp))
        self.assertTrue(np.allclose(delta_t_vals, delta_t_vals_comp))

        self.assertTrue(np.allclose(kernel_dims, kernel_dims_comp))

        #check that data is equal to data taken from a single kernel (all kernels in the array are equivalent here)
        self.assertTrue(np.all([val == self.kernel.eps for val in errors]))
        self.assertTrue(np.all([val == self.kernel.m for val in m_vals]))
        self.assertTrue(np.all([val == self.kernel.n for val in n_vals]))
        self.assertTrue(np.all([val == self.kernel.h for val in h_vals]))
        self.assertTrue(np.all([val == self.kernel.ID_rank for val in ID_ranks]))
        self.assertTrue(np.all([val == self.kernel.beta for val in betas]))
        self.assertTrue(np.all([val == self.kernel.delta_t for val in delta_t_vals]))
        self.assertTrue(np.all([val == self.kernel.N_max for val in N_maxs]))
        self.assertTrue(np.allclose(kernel_dims, np.array([2,2])))

    

    def tearDown(self):
        # This method runs after each test and cleans up temporarily created hdf5 files that are no longer needed
        if os.path.exists(self._filename):
            os.remove(self._filename)

    
    if __name__ == "__main__":
        unittest.main()
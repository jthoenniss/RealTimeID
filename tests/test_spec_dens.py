import unittest
import numpy as np
from src.spec_dens.spec_dens import spec_dens_gapless, spec_dens_gapped_sym


class Test_spec_dens_gapless(unittest.TestCase):

    def setUp(self):
        # define test parameters
        self.cutoff_lower = -100.
        self.cutoff_upper = 100.
        self.Gamma = 1.0
        self.spec_dens = lambda x: spec_dens_gapless(x, cutoff_lower= self. cutoff_lower, cutoff_upper = self.cutoff_upper, Gamma = self.Gamma)
    
    def test_spec_dens_gapless_vectorized(self):
        """
        Test the vectorized implementation of the spectral density of the gapped model.
        """
        #______Define frequency grids for different frequency regions______
        #above the bulk
        omegas_above_cutoff = np.linspace( self.cutoff_upper + 10, self.cutoff_upper + 100 , 10)
        #below the bulk
        omegas_below_cutoff = np.linspace(self.cutoff_lower - 100 , self.cutoff_lower - 10, 10)
        #in the bulk
        omegas_between_cutoffs = np.linspace(self.cutoff_lower + 10, self.cutoff_upper - 10, 10)
        
        #_________Test implementation of spectral density_________
        #above the bulk
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_above_cutoff),0. , rtol=1e-4)))
        #below the bulk
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_below_cutoff),0. , rtol=1e-4)))
        # check if spectral density is Gamma between cutoffs
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_between_cutoffs), self.Gamma , rtol=1e-4)))

    

    def test_spec_dens_scalar(self):
        """
        Test the implementation of the spectral density of the gapped model for scalar frequency arguments
        """
        #in the center of the bulk
        self.assertTrue(np.isclose(self.spec_dens(self.cutoff_lower + (self.cutoff_upper - self.cutoff_lower) / 2 ), self.Gamma, rtol=1e-4))
        #below the bulk
        self.assertTrue(np.isclose(self.spec_dens(self.cutoff_lower - 10), 0., rtol=1e-4))
        #above the bulk
        self.assertTrue(np.isclose(self.spec_dens(self.cutoff_upper + 10), 0., rtol=1e-4))
    


class Test_spec_dens_gapped(unittest.TestCase):

    def setUp(self):
        # define test parameters
        self.cutoff_lower = 20.#lower cutoff frequency for positive frequencies (defines upper cutoff of the gap)
        self.cutoff_upper = 100. #upper cutoff frequency for positive frequencies
        self.Gamma = 1.0
        self.spec_dens = lambda x: spec_dens_gapped_sym(x, cutoff_lower= self.cutoff_lower, cutoff_upper = self.cutoff_upper, Gamma = self.Gamma)
    
    def test_spec_dens_gapped_vectorized(self):
        """
        Test the vectorized implementation of the spectral density of the gapped model.
        """
        
        #______Define frequency grids for different frequency regions______
        #above the upper bulk
        omegas_above_upper_bulk = np.linspace( self.cutoff_upper + 10, self.cutoff_upper + 100 , 10)
        #in the upper bulk
        omegas_in_upper_bulk = np.linspace( self.cutoff_lower + 10 , self.cutoff_upper - 10, 10)
        #in the gap (below upper bulk and above lower bulk)
        omegas_in_gap = np.linspace(- self.cutoff_lower + 10, self.cutoff_lower - 10, 10)
        #in the lower bulk
        omegas_in_lower_bulk = np.linspace( -self.cutoff_upper + 10 , -self.cutoff_lower - 10, 10)
        #below the lower bulk
        omegas_below_lower_bulk = np.linspace( -self.cutoff_upper -100 , -self.cutoff_upper - 10 , 10)
        
        #_________Test implementation of spectral density_________  
        #check if spectral density is zero above upper bulk
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_above_upper_bulk), 0. , rtol=1e-4)))
        #check if spectral density is Gamma below upper bulk
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_in_upper_bulk), self.Gamma , rtol=1e-4)))
        #check if spectral density is zero in gap
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_in_gap), 0. , rtol=1e-4)))
        #check is spectral density is Gamma in lower bulk
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_in_lower_bulk), self.Gamma , rtol=1e-4)))
        #check if spectral density is zero below lower bulk
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_below_lower_bulk), 0. , rtol=1e-4)))

if __name__ == "__main__":
    unittest.main()
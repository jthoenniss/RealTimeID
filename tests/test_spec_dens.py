import unittest
import numpy as np
from src.spec_dens.spec_dens import spec_dens_gapless


class Test_spec_dens_gapless(unittest.TestCase):

    def setUp(self):
        # define test parameters
        self.cutoff_lower = -10.
        self.cutoff_upper = 10.
        self.Gamma = 1.0
        self.spec_dens = lambda x: spec_dens_gapless(x, cutoff_lower= self. cutoff_lower, cutoff_upper = self.cutoff_upper, Gamma = self.Gamma)
    
    def test_spec_dens_gapless_vectorized(self):
        """
        Test the vectorized implementation of the spectral density of the gapped model.
        """
        
        
        omegas_above_cutoff = np.linspace(1.2 * self.cutoff_upper, 5 * self.cutoff_upper , 10)
        omegas_below_cutoff = np.linspace(5 * self.cutoff_lower , 1.2 * self.cutoff_lower, 10)
        omegas_between_cutoffs = np.linspace(0.8 * self.cutoff_lower, 0.8 * self.cutoff_upper, 10)
        

        # check if spectral density is zero beyond cutoffs
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_above_cutoff),0. , rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_below_cutoff),0. , rtol=1e-10)))
        # check if spectral density is Gamma between cutoffs
        self.assertTrue(np.all(np.isclose(self.spec_dens(omegas_between_cutoffs), self.Gamma , rtol=1e-10)))

    

    def test_spec_dens_scalar(self):
        """
        Test the implementation of the spectral density of the gapped model for scalar frequency arguments
        """
        
        self.assertTrue(np.isclose(self.spec_dens(0), self.Gamma, rtol=1e-10))
        self.assertTrue(np.isclose(self.spec_dens(1.2 * self.cutoff_lower), 0., rtol=1e-10))
        self.assertTrue(np.isclose(self.spec_dens(1.2 * self.cutoff_upper), 0., rtol=1e-10))
    
if __name__ == "__main__":
    unittest.main()
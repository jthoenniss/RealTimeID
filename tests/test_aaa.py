import unittest
import numpy as np
import os, sys
project_path = os.environ.get('REALTIMEID_PATH')
if project_path and project_path not in sys.path:
    sys.path.append(project_path)
    print("Project path successfully added.")

from src.AAA.AAA_kernel import AAARep
from src.AAA.aaa_algorithm import aaa, cleanup
from src.spec_dens.spec_dens import spec_dens_semi_circle
from src.utils.common_funcs import initialize_fine_grid, set_time_grid

class TestAAARep(unittest.TestCase):
    def setUp(self):

        self.params = {
            "m": 10,
            "n": 5,
            "beta": 1.0,
            "N_max": 10,
            "delta_t": 0.1,
            "h": 0.2,
            "phi": np.pi / 4,
            "upper_cutoff" : 600,
            "spec_dens": lambda x: spec_dens_semi_circle(x),
            "freq_parametrization": "simple_exp",
        }

        self.K = AAARep(**self.params)

    def test_init(self):
        print("Test initialization of AAARep object.")
        self.assertEqual(self.K.m, 10)
        self.assertEqual(self.K.n, 5)
        self.assertEqual(self.K.beta, 1.0)
        self.assertEqual(self.K.h, 0.2)
        self.assertEqual(self.K.freq_parametrization, "simple_exp")

    
    def test_grids(self):
        print("Test AAA algorithm.")
        
        self.assertEqual(self.K.Z.size, 2 * (self.K.m + self.K.n + 1))
        self.assertEqual(self.K.F_particle.size, 2 * (self.K.m + self.K.n + 1))
        self.assertEqual(self.K.F_hole.size, 2 * (self.K.m + self.K.n + 1))


        #explicitly get the fine frequency grid
        fine_grid, _, _ = initialize_fine_grid(self.params["m"], self.params["n"], self.params["h"], freq_parametrization= "simple_exp")
        # frequency points in for negativ and positive part on real axis
        Z = np.concatenate((-fine_grid[::-1], fine_grid)) 
        self.assertTrue(np.allclose(self.K.Z, Z))

        #create function values by hand:
        # function values, determined by product of spectral density and Fermi Dirac distribution and Jacobian factor h*abs(x)
        def func_exact(x):
            fd_particle = 1/(1 + np.exp(-self.params["beta"] * x)) 
            f_particle = self.params["spec_dens"](x) * fd_particle
         
            fd_hole = 1/(1 + np.exp(self.params["beta"] * x))
            f_hole = self.params["spec_dens"](x) * fd_hole
            
            return f_particle, f_hole
        
        F_particle, F_hole = func_exact(Z)
        

        #compare to values in object
        self.assertTrue(np.array_equal(self.K.F_particle, F_particle))
        self.assertTrue(np.allclose(self.K.F_hole, F_hole))

        #manually perform AAA algorithm on spectral density multiplied with Fermi-Dirac distribution
        r_particle, errors_particle = aaa(Z = Z, F = F_particle, return_errors=True)
        r_hole, errors_hole = aaa(Z = Z, F = F_hole, return_errors=True)

        #compare to values in object
        #errors
        self.assertTrue(np.allclose(self.K.errors_particle, errors_particle), f"Errors do not coincide, got {self.K.errors_particle} and {errors_particle}")
        self.assertTrue(np.allclose(self.K.errors_hole, errors_hole), f"Errors do not coincide, got {self.K.errors_hole} and {errors_hole}")

        #poles and residues
        self.assertTrue(np.allclose(self.K.r_particle.polres(), r_particle.polres()), f"Poles and residues do not coincide, got {self.K.r_particle.polres()} and {r_particle.polres()}")
        self.assertTrue(np.allclose(self.K.r_hole.polres(), r_hole.polres()), f"Poles and residues do not coincide, got {self.K.r_hole.polres()} and {r_hole.polres()}")

        #remove Froissart doublets and repeat all checks
        r_particle = cleanup(r_particle, Z, F_particle)
        r_hole = cleanup(r_hole, Z, F_hole)
        self.K.remove_Froissart()

        #errors
        self.assertTrue(np.allclose(self.K.errors_particle, errors_particle), f"Errors do not coincide, got {self.K.errors_particle} and {errors_particle}")
        self.assertTrue(np.allclose(self.K.errors_hole, errors_hole), f"Errors do not coincide, got {self.K.errors_hole} and {errors_hole}")

        #poles and residues
        self.assertTrue(np.allclose(self.K.r_particle.polres(), r_particle.polres()), f"Poles and residues do not coincide, got {self.K.r_particle.polres()} and {r_particle.polres()}")
        self.assertTrue(np.allclose(self.K.r_hole.polres(), r_hole.polres()), f"Poles and residues do not coincide, got {self.K.r_hole.polres()} and {r_hole.polres()}")


    def test_polres(self):
        print("test poles and residues from AAA")

        #get poles and residues
        polres_particle, polres_hole = self.K.polres() 
        polres_particle_explicit = self.K.r_particle.polres()
        polres_hole_explicit = self.K.r_hole.polres()

        #check that particles and holes are returned correctl in polres()
        self.assertTrue(np.allclose(polres_particle, polres_particle_explicit))
        self.assertTrue(np.allclose(polres_hole, polres_hole_explicit))

    def test_propag(self):
        print("test propagator from AAA")

        #get poles and residues
        polres_particle, polres_hole = self.K.polres() 

        #identify poles in upper half plane and corresponding residues
        poles_particle_upper = polres_particle[0][np.imag(polres_particle[0]) > 0]
        residues_particle_upper = polres_particle[1][np.imag(polres_particle[0]) > 0]

        poles_hole_upper = polres_hole[0][np.imag(polres_hole[0]) > 0]
        residues_hole_upper = polres_hole[1][np.imag(polres_hole[0]) > 0]

        #define time gid
        times = set_time_grid(N_max=self.params["N_max"], delta_t=self.params["delta_t"])
        #sum over all modes expliclity
        G_particle = 2.j * np.pi * np.sum(residues_particle_upper  * np.exp(1.j * poles_particle_upper * times[:,np.newaxis]), axis=1).flatten()
        G_hole = 2.j * np.pi * np.sum(residues_hole_upper  * np.exp(1.j * poles_hole_upper * times[:,np.newaxis]), axis=1).flatten()

        #concatenate particle and hole propagator
        G = np.concatenate((G_particle, G_hole))
        #compare to values in object
        self.assertTrue(np.allclose(G, self.K.propagator_AAA(times)))

if __name__ == "__main__":
    unittest.main()
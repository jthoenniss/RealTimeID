import unittest
import numpy as np
import tensorflow as tf

from src.lindblad_optimization.local_lindblad_optimization import propag_from_params


class test_propag_from_params(unittest.TestCase):

    def setUp(self):
        # consider only two modes, whose parameters are given by the following:
        # (columns correspond to different modes, rows to different parameters: Gamma, omega, and gamma)
        Gamma_1, Gamma_2 = 1., 2.
        omega_1, omega_2 = 3., 4.
        gamma_1, gamma_2 = 5., 6.
        

        self.parameters_tf = tf.Variable([[Gamma_1, Gamma_2], [omega_1, omega_2], [gamma_1, gamma_2]], dtype=tf.float64)

        # array of time points
        self.time_grid = [0, 1, 2, 3, 4, 5]
        self.time_grid_np = np.array(self.time_grid)
        self.time_grid_tf = tf.cast(tf.expand_dims(self.time_grid, -1), tf.float64)  # Equivalent to[:, np.newaxis] in numpy

        # expected propagator. Compute separately for both modes
        propag_mode1 = Gamma_1 * np.exp(
            (1.0j * omega_1 - gamma_1)
            * self.time_grid_np
        )
        propag_mode2 = Gamma_2 * np.exp(
            (1.0j * omega_2 - gamma_2)
            * self.time_grid_np
        )


        # create matrix where each column corresponds to the propagator of a mode:
        propagmatrix = np.hstack(
            [propag_mode1[:, np.newaxis], propag_mode2[:, np.newaxis]]
        )  # shape (6, 2)

        # sum over the columns to get the expected propagator:
        self.expected_propag = np.sum(propagmatrix, axis=1)

    def test_propag_from_params(self):
        
        propag = propag_from_params(self.parameters_tf, self.time_grid_tf) 
        # check that the computed propagator is close to the expected propagator
        self.assertTrue(np.allclose(propag, self.expected_propag))


if __name__ == "__main__":
    unittest.main()

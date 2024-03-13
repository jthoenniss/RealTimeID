import unittest
import numpy as np
import tensorflow as tf

from src.lindblad_optimization.local_lindblad_optimization import propag_from_params, custom_loss


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


class test_custom_loss(unittest.TestCase):

    def setUp(self) -> None:
        
        propag1 = [1 + 0.5*1.j, 2+ 0.5*1.j, 3+ 0.5*1.j, 4+ 0.5*1.j, 5+ 0.5*1.j]
        propag2 = [1 + 0.5*3.j, 4+ 0.5*6.j, 1+ 0.5*0.j, 2+ 0.5*6.j, 8+ 0.5*8.j]

        self.propag1_tf = tf.constant(propag1, dtype=tf.complex128)
        self.propag2_tf = tf.constant(propag2, dtype=tf.complex128)

        self.error_np = np.mean(np.abs(np.array(propag1) - np.array(propag2)) ** 2)

    
    def test_custom_loss(self):
        error_tf = custom_loss(self.propag1_tf, self.propag2_tf)
        self.assertTrue(np.allclose(error_tf, self.error_np), f"Error: {error_tf} is not close to expected error: {self.error_np}")


if __name__ == "__main__":
    unittest.main()

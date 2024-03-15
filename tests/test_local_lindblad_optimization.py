import unittest
import numpy as np
import tensorflow as tf

from src.lindblad_optimization.local_lindblad_optimization import propag_from_params, custom_loss


class test_propag_from_params(unittest.TestCase):

    def setUp(self):
        # consider only two modes, whose parameters are given by the following:
        # (columns correspond to different modes, rows to different parameters: Gamma, omega, and gamma)
        Gamma_k = [1., 2.]# Gamma_1, Gamma_2 = 1., 2.
        omega_k = [3., 4.]# omega_1, omega_2 = 3., 4.
        gamma_k = [5., 6.]# gamma_1, gamma_2 = 5., 6.
    

        self.Gamma_k_tf = tf.constant(Gamma_k, dtype=tf.float64)
        self.omega_k_tf = tf.constant(omega_k, dtype=tf.float64)
        self.gamma_k_tf = tf.constant(gamma_k, dtype=tf.float64)

        # array of time points
        self.time_grid = [0, 1, 2, 3, 4, 5]
        self.time_grid_np = np.array(self.time_grid)

        self.time_grid_tf = tf.cast(tf.expand_dims(self.time_grid, -1), tf.float64)  # Equivalent to[:, np.newaxis] in numpy

        # expected propagator. Compute separately for both modes
        propag_mode1 = Gamma_k[0] * np.exp(
            (1.0j * omega_k[0] - gamma_k[0])
            * self.time_grid_np
        )
        propag_mode2 = Gamma_k[1] * np.exp(
            (1.0j * omega_k[1] - gamma_k[1])
            * self.time_grid_np
        )


        # create matrix where each column corresponds to the propagator of a mode:
        propagmatrix = np.hstack(
            [propag_mode1[:, np.newaxis], propag_mode2[:, np.newaxis]]
        )  # shape (6, 2)

        # sum over the columns to get the expected propagator:
        self.expected_propag = np.sum(propagmatrix, axis=1)

    def test_propag_from_params(self):
        
        propag = propag_from_params(self.time_grid_tf, self.Gamma_k_tf, self.omega_k_tf, self.gamma_k_tf).numpy() # this returns a numpy array with shape (2,n_timesteps), where the first row represents the real values and the second row the imaginary values of the
        
        #check dimensions of propag which should be a 2x6 tensor flow tensor:
        self.assertEqual(propag.shape, (2,6), f"Shape of propag: {propag.shape} is not equal to (2,6)")


        propag_complex = propag[0] + 1.j * propag[1]  # convert to complex numbers
        # check that the computed propagator is close to the expected propagator
        self.assertTrue(np.allclose(propag_complex, self.expected_propag), f"Computed propagator: {propag_complex} is not close to expected propagator: {self.expected_propag}")


class test_custom_loss(unittest.TestCase):

    def setUp(self) -> None:
        time_grid = [1.,2.,3.,4.,5.,6.]
        self.time_grid_tf = tf.cast(tf.expand_dims(time_grid, -1), tf.float64)  # Equivalent to[:, np.newaxis] in numpy

        self.propag_reference = propag_from_params(time_grid=self.time_grid_tf,Gamma_k=tf.constant([1., 2.], dtype= tf.float64), omega_k=tf.constant([3., 4.], dtype= tf.float64), gamma_k=tf.constant([5., 6.], dtype= tf.float64) )
        self.propag_detuned = propag_from_params(time_grid=self.time_grid_tf, Gamma_k=tf.constant([1.2, 2.2], dtype= tf.float64), omega_k=tf.constant([3.2, 4.2], dtype= tf.float64), gamma_k=tf.constant([5.2, 6.2], dtype= tf.float64) )
  
        #compute the loss function explicitly using numpy commands
        self.loss_numpy = np.mean(np.abs(self.propag_reference.numpy() - self.propag_detuned.numpy())**2)

    
    def test_custom_loss(self):
  
        #check dimensions of propag which should be a 2x6 tensor flow tensor:
        self.assertEqual(self.propag_reference.shape, (2,6), f"Shape of propag: {self.propag_reference.shape} is not equal to (2,6)")
        self.assertEqual(self.propag_detuned.shape, (2,6), f"Shape of propag: {self.propag_detuned.shape} is not equal to (2,6)")

        #compute the loss function using the custom_loss function
        loss_tf = custom_loss(self.propag_reference, self.propag_detuned)
        # check that the computed loss is close to the expected loss
        self.assertTrue(np.allclose(loss_tf, self.loss_numpy), f"Error: {loss_tf} is not close to expected error: {self.loss_numpy}")


if __name__ == "__main__":
    unittest.main()

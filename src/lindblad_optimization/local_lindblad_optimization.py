import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# define the propagator function that takes a list of parameters and a time grid and returns the propagator
def propag_from_params(parameters: tf.Tensor, time_grid: tf.Tensor) -> tf.Tensor:
    """
    Calculates the propagator vector based on the given parameters and time grid.

    Args:
        - parameters (tf.Tensor): A tensor containing the parameters for the calculation.
            First row: Gamma_k (coupling), 
            second row: omega_k (unitary evolution), 
            third row: gamma_k (decay).

        - time_grid (tf.Tensor): A tensor representing the time grid.

    Returns:
        tf.Tensor: The calculated propagator vector.
    """
  
    Gamma_k = tf.cast(parameters[0, :], tf.complex64)
    omega_k = tf.cast(parameters[1, :], tf.complex64)
    gamma_k = tf.cast(parameters[2, :], tf.complex64)

    #enable broadcasting by adding a dimension to the time grid and convert to complex data type
    time_grid_broad = tf.cast(tf.expand_dims(time_grid, -1), tf.complex64)  # Equivalent to[:, np.newaxis] in numpy

    propag_mat = Gamma_k * tf.exp((1.j * omega_k - gamma_k) * time_grid_broad)
    propag_vec = tf.reduce_sum(propag_mat, axis=1)

    return propag_vec



# Define the propagator function that takes a list of parameters and a time grid and returns the propagator
class PropagLayer(tf.keras.layers.Layer):
    """ 
    Custom layer that computes the propagator from a list of parameters and a time grid.
    This is convenient because it allows us to use the keras API to optimize the parameters.
    """

    def __init__(self, time_grid, **kwargs):
        #initialize the layer with the time grid
        super(PropagLayer, self).__init__(**kwargs)
        self.time_grid = time_grid

    def call(self, parameters):
        # Convert python function using numpy to tensorflow function in order to use automatic differentiation
        # (Automatic differentiation is only possible for tensorflow functions, not numpy functions.)
        return tf.numpy_function(propag_from_params, [parameters, self.time_grid], Tout=tf.float32)

# Define the model
class PropagModel(tf.keras.Model):
    """
    Custom model that includes the PropagLayer. The parameters are the only trainable variables in the model.
    """
    def __init__(self, time_grid, initial_params):
        #initialize the model with the time grid and the initial parameters
        super(PropagModel, self).__init__()
        # Initialize the propagator layer
        self.propag_layer = PropagLayer(time_grid)
        # Initialize parameters as a trainable variable
        self.params = tf.Variable(initial_params, dtype=tf.float32)

    def call(self, x):
        # Call the propagator layer with the parameters
        return self.propag_layer(self.params)
    
# Loss function: Sum of squared errors
def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Function to compute the loss between the target vector and the predicted vector.
    Parameters:
    - y_true: tf.Tensor, the target vector
    - y_pred: tf.Tensor, the predicted vector

    Returns:
    - tf.Tensor: the loss between the two vectors
    """
    return tf.reduce_sum(tf.square(y_true - y_pred))



if __name__== "__main__":

    # Define your target vector and time grid
    target_vector = np.array([0.,0.,0.]) 
    time_grid = np.array([1.,2.,3.])
    initial_params = np.array([1.,1.,1.]) 

    # Instantiate model
    model = PropagModel(time_grid, initial_params)


    # Compile the model
    model.compile(optimizer='adam', loss=custom_loss)

    # Since the model technically doesn't "fit" to data in the traditional sense,
    # we use a dummy input (x) and the target vector as y.
    # The actual optimization occurs with respect to the parameters within the model.
    model.fit(x=np.zeros((1, 1)), y=target_vector.reshape(1, -1), epochs=100)
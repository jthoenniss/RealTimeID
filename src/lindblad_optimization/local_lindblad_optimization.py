import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# define the propagator function that takes a list of parameters and a time grid and returns the propagator
def propag_from_params(parameters: np.ndarray, time_grid) -> np.ndarray:
    """
    Function that yields the propagator g(t) = \sum_k Gamma(k) * exp((1j * omega - gamma) * t)

    Parameters:
    List of dictionaries containing the parameters for each mode. Each dictionary contains the following
    keys:
        - Gamma: float (encoding cupling)
        - omega: float (encoding unitary evolution)
        - gamma: float (encoding decay)

    Returns:
    np.ndarray: the propagator g(t) for the given parameters and time grid
    """
    # collect every parameter type for all modes in a numpy array, respectively.
    Gamma_k = np.array([params['Gamma'] for params in parameters]) 
    omega_k = np.array([params['omega'] for params in parameters])
    gamma_k = np.array([params['gamma'] for params in parameters])

    #broadcast time grid to match the shape of the parameters
    time_grid_broad = time_grid[:, np.newaxis] 

    # initialize the propagator matrix where rows are time steps and columns are modes
    propag_mat = Gamma_k * np.exp((1j * omega_k - gamma_k) * time_grid_broad)

    # sum over all modes and flatten array:
    propag_vec = np.sum(propag_mat, axis=1).flatten()# Resulting shape: (N_timesteps,)

    #return propagator
    return propag_vec


# define the error function
def error_function(propag1: np.ndarray, propag2:np.ndarray) -> float:
    """
    Function to compute the error between two propagators. The error is the sum of the squared differences at each time point.
    Parameters:
    - propag1: np.ndarray, the first propagator
    - propag2: np.ndarray, the second propagator

    Returns:
    - float: the error between the two propagators
    """

    # compute the error
    error = np.sum(np.abs(propag1 - propag2)**2)

    return error

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
        #convert python function using numpy to tensorflow function:
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

# Define your target vector and time grid
target_vector = np.array([...]) 
time_grid = np.array([...])
initial_params = np.array([...]) 

# Instantiate model
model = PropagModel(time_grid, initial_params)

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

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)

# Since the model technically doesn't "fit" to data in the traditional sense,
# we use a dummy input (x) and the target vector as y.
# The actual optimization occurs with respect to the parameters within the model.
model.fit(x=np.zeros((1, 1)), y=target_vector.reshape(1, -1), epochs=100)
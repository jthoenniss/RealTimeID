import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# define the propagator function that takes a list of parameters and a time grid and returns the propagator
def propag_from_params(parameters: np.ndarray, time_grid) -> np.ndarray:
    """
    Vectorized function that yields the propagator g(t) = \sum_k Gamma(k) * exp((1j * omega - gamma) * t)

    Parameters:
    - parameters: np.ndarray, shape (3, N_modes), where N_modes is the number of modes. 
        The first row contains the Gamma parameters describing coupling strength, 
        the second row contains the omega parameters describing unitary evolution, 
        and the third row contains the gamma parameters, describing decay.
    
    Returns:
    np.ndarray: the propagator g(t) for the given parameters and time grid
    """
    
    Gammas = parameters[0] # shape (N_modes,). Encoding coupling strength
    omegas = parameters[1] # shape (N_modes,). Encoding unitary evolution
    gammas = parameters[2] # shape (N_modes,). Encoding decay

    # Use broadcasting to create a (N_timesteps, N_modes) shaped array for time-dependent exponent
    exponent = (1j * omegas - gammas) * time_grid[:, np.newaxis]

    # Calculate the full propagator matrix
    propag_mat = Gammas * np.exp(exponent)

    # Sum over all modes to get the final propagator vector
    propag_vec = np.sum(propag_mat, axis=1)

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


"""if __name__== "__main__":
    # test the function propag_from_params
    parameters = [{'Gamma': 1, 'omega': 1, 'gamma': 1}, {'Gamma': 2, 'omega': 2, 'gamma': 2}]
    time_grid = np.array([0, 1, 2, 3, 4, 5])
    propag = propag_from_params(parameters, time_grid)
    expected_propag = """
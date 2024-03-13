import numpy as np
import tensorflow as tf



# define the propagator function that takes a list of parameters and a time grid and returns the propagator
def propag_from_params(parameters: tf.Tensor, time_grid: tf.Tensor) -> tf.Tensor:
    assert time_grid.shape[1] == 1, "Time grid must be a column vector (i.e., have shape (n, 1))."

    Gamma_k = parameters[0]  # Gamma_k is used as real in complex multiplication later.
    omega_k = parameters[1]
    gamma_k = parameters[2]

    # First, ensure omega_k and gamma_k are cast to complex
    omega_k_complex = tf.cast(omega_k, tf.complex128)
    gamma_k_complex = tf.cast(gamma_k, tf.complex128)
    time_grid_complex = tf.cast(time_grid, tf.complex128)

    # Now, perform the complex arithmetic with all operands as complex tensors
    complex_exponent = 1j * omega_k_complex - gamma_k_complex
    propag_mat = tf.cast(Gamma_k, tf.complex128) * tf.exp(complex_exponent * time_grid_complex)
    propag_vec = tf.reduce_sum(propag_mat, axis=1)

    return tf.reshape(propag_vec, (-1,))# return flattened vector




# Define the propagator function that takes a list of parameters and a time grid and returns the propagator
class PropagLayer(tf.keras.layers.Layer):
    """ 
    Custom layer that computes the propagator from a list of parameters and a time grid.
    This is convenient because it allows us to use the keras API to optimize the parameters:
    The parameters are the only trainable variables in the model.
    """

    def __init__(self, time_grid, **kwargs):
        """
        Initialize the layer with the time grid.
        Args:
        - time_grid (array): The time grid on which the propagator is calculated. Will be cast to a tf.Tensor of shape (n, 1).
        - **kwargs: Additional keyword arguments for the layer.

        Returns:
        - None
        """
        super(PropagLayer, self).__init__(**kwargs)
        self.time_grid = tf.cast(tf.expand_dims(time_grid, -1), tf.float64)  # converted to tensor flow column vector. Analogue to[:, np.newaxis] in numpy

    def call(self, parameters):
        # Convert python function using numpy to tensorflow function in order to use automatic differentiation
        # (Automatic differentiation is only possible for tensorflow functions, not numpy functions.)
        return propag_from_params(parameters, self.time_grid)

# Define the model
class PropagModel(tf.keras.Model):
    """
    Custom model that includes the PropagLayer. The parameters are the only trainable variables in the model.
    """
    def __init__(self, time_grid, initial_params):
        """
        Initialize the model with the time grid and the initial parameters.
        Args:
        - time_grid (array): The time grid on which the propagator is calculated.
        - initial_params (array): The initial parameters for the model. 
            First row: Gamma_k (coupling), 
            second row: omega_k (unitary evolution), 
            third row: gamma_k (decay).
        Returns:
        - None
        """
        #initialize the model with the time grid and the initial parameters
        super(PropagModel, self).__init__()
        # Initialize the propagator layer
        self.propag_layer = PropagLayer(time_grid)
        # Initialize parameters as a trainable variable
        self.params = tf.Variable(initial_params, dtype=tf.float64, trainable=True) 


    def call(self, x):
        # Call the propagator layer with the parameters
        return self.propag_layer(self.params)
    
    
# Loss function: Sum of squared errors
def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Function to compute the mean squared error between the target vector and the predicted vector.
    Parameters:
    - y_true: tf.Tensor, the target vector
    - y_pred: tf.Tensor, the predicted vector

    Returns:
    - tf.Tensor: the loss between the two vectors (as real, positive number), [sum_i (|(y_true)_i - (y_pred)_i|)^2 ] / n, where n is the length of the vectors.
    """
    return tf.losses.mean_squared_error(y_true, y_pred)


class PrintParametersCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Assuming 'params' is the attribute holding the parameters you're optimizing.
        # Adjust this to match how your model stores its parameters.
        parameters = self.model.params
        print(f"\n Parameters after epoch {epoch+1}: {parameters} \n")

        logs = logs or {}
        error = logs.get('loss')
        print(f"Error at epoch {epoch}: {error}")


if __name__== "__main__":


    Gamma_1, Gamma_2 = 1., 2.
    omega_1, omega_2 = 3., 4.
    gamma_1, gamma_2 = 5., 6.
    

    parameters = np.array([[Gamma_1, Gamma_2], [omega_1, omega_2], [gamma_1, gamma_2]])

    # array of time points

    time_grid_np = np.arange(0.1, 10., .05) * np.pi
    time_grid = time_grid_np.tolist()
 
    # expected propagator. Compute separately for both modes
    propag_mode1 = Gamma_1 * np.exp(
        (1.0j * omega_1 - gamma_1)
        * time_grid_np
    )
    propag_mode2 = Gamma_2 * np.exp(
        (1.0j * omega_2 - gamma_2)
        * time_grid_np
    )

    # create matrix where each column corresponds to the propagator of a mode:
    propagmatrix = np.hstack(
        [propag_mode1[:, np.newaxis], propag_mode2[:, np.newaxis]]
    )  # shape (6, 2)

    # sum over the columns to get the expected propagator:
    expected_propag = np.sum(propagmatrix, axis=1)
    
    

    # Define your target vector and time grid
    target_vector = tf.constant(expected_propag, dtype=tf.complex128)
   
    #generate random  array of length 100:
    Gamma_inits = np.abs(np.random.rand(1000).tolist())
    omega_inits = np.random.rand(1000).tolist()
    gamma_inits = np.abs(np.random.rand(1000).tolist())

    
    initial_params = [Gamma_inits, omega_inits, gamma_inits]

    
    # Create an instance of your custom callback
    #print_parameters_callback = PrintParametersCallback()

    # Instantiate model
    model = PropagModel(time_grid, initial_params)
    
     
    
    # Before your model compilation, specify a custom learning rate
    custom_learning_rate = 0.001  # This is an example value; adjust based on your needs

    # Instantiate the Adam optimizer with the custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)

    # Compile the model with the custom optimizer
    model.compile(optimizer=optimizer, loss=custom_loss)

    x_dummy = tf.zeros((len(time_grid),), dtype=tf.complex64)
  
    # Since the model technically doesn't "fit" to data in the traditional sense,
    # we use a dummy input (x) and the target vector as y.
    # The actual optimization occurs with respect to the parameters within the model.
    # and specify callback
    model.fit(x=x_dummy, y=target_vector, epochs=10000, callbacks=[PrintParametersCallback()])



   
    print("output propag", model(tf.zeros((len(time_grid),), dtype=tf.complex64)))
    print("target_vector", target_vector)   
    got = model(tf.zeros((len(time_grid),), dtype=tf.complex64))
    print("error",tf.losses.mean_squared_error(target_vector, got))
  
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# define the propagator function that takes a list of parameters and a time grid and returns the propagator
def propag_from_params(time_grid: tf.Tensor, Gamma_k: tf.Tensor, omega_k: tf.Tensor, gamma_k: tf.Tensor) -> tf.Tensor:
    assert time_grid.shape[1] == 1, f"Time grid must be a column vector (i.e., have shape (n, 1)), but detected shape is {time_grid.shape}."

    real_part = tf.reduce_sum(tf.exp(-gamma_k * time_grid) * (Gamma_k * tf.cos(omega_k * time_grid)), axis=1)
    imag_part = tf.reduce_sum(tf.exp(-gamma_k * time_grid) * (Gamma_k * tf.sin(omega_k * time_grid)), axis=1)

    #flatten both arrays and stack them to get a 2D tensor
    real_part = tf.reshape(real_part, (-1,))
    imag_part = tf.reshape(imag_part, (-1,))

    #stack real and imaginary part in (2, len(time_grid)) tensor
    propag_vec_real_valued = tf.stack([real_part, imag_part], axis=0)

    return propag_vec_real_valued


class PropagModel(tf.keras.Model):
    def __init__(self, time_grid: list, initial_Gamma_k: list, initial_omega_k: list, initial_gamma_k: list):
        super(PropagModel, self).__init__()

        self.time_grid = tf.reshape(tf.constant(time_grid, dtype=tf.float32), (-1, 1))# Ensure time grid is a column vector
        
        # Directly add constrained parameters to the model
        self.Gamma_k = self.add_weight(name='Gamma_k',
                                       shape=(len(initial_Gamma_k),),
                                       initializer=tf.keras.initializers.Constant(initial_Gamma_k),
                                       constraint=tf.keras.constraints.NonNeg(),
                                       dtype=tf.float32,
                                       trainable=True)
        
        self.omega_k = self.add_weight(name='omega_k',
                                       shape=(len(initial_omega_k),),
                                       initializer=tf.keras.initializers.Constant(initial_omega_k),
                                       dtype=tf.float32,
                                       trainable=True)
        
        self.gamma_k = self.add_weight(name='gamma_k',
                                       shape=(len(initial_gamma_k),),
                                       initializer=tf.keras.initializers.Constant(initial_gamma_k),
                                       constraint=tf.keras.constraints.NonNeg(),
                                       dtype=tf.float32,
                                       trainable=True)

    def call(self, inputs):
        return propag_from_params(self.time_grid, self.Gamma_k, self.omega_k, self.gamma_k)
    
    def target_vector(self, target_Gamma_k: tf.Tensor, target_omega_k: tf.Tensor, target_gamma_k: tf.Tensor) -> tf.Tensor:
        return propag_from_params(self.time_grid, target_Gamma_k, target_omega_k, target_gamma_k)
    
    
def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the mean squared error between y_true and y_pred, 
  
    Parameters:
    - y_true: TensorFlow tensor containing the true values.
    - y_pred: TensorFlow tensor containing the predicted values.
   
    Returns:
    - A tensor containing the computed loss.
    """
    
    # Compute the combined absolute values of the filtered values
    difference = y_true - y_pred

    mse_loss = tf.reduce_mean(tf.square(tf.abs(difference)))


    return mse_loss

class PrintParametersCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Adjust this to match how your model stores its parameters.
    
        logs = logs or {}
        error = logs.get('loss')
        print(f"Error at epoch {epoch}: {error}")

class RecordParametersCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(RecordParametersCallback, self).__init__()
        # Initialize a dictionary to hold lists of parameter values
        self.parameters_over_time = {}
        
    def on_epoch_end(self, epoch, logs=None):
        # Record the parameters of interest
        # For example, to record the value of 'Gamma_k'
        if 'Gamma_k' not in self.parameters_over_time:
            self.parameters_over_time['Gamma_k'] = []
        if 'omega_k' not in self.parameters_over_time:
            self.parameters_over_time['omega_k'] = []
        if 'gamma_k' not in self.parameters_over_time:
            self.parameters_over_time['gamma_k'] = []


        self.parameters_over_time['Gamma_k'].append(self.model.Gamma_k.numpy())
        self.parameters_over_time['omega_k'].append(self.model.omega_k.numpy())
        self.parameters_over_time['gamma_k'].append(self.model.gamma_k.numpy())




if __name__== "__main__":

    nbr_epochs = 100

    #define time grid and parameters for target vector
    # consider only two modes, whose parameters are given by the following:
    # (columns correspond to different modes, rows to different parameters: Gamma, omega, and gamma)
    Gamma_1, Gamma_2 = 1.31, 2.31
    omega_1, omega_2 = 3.31, 4.31
    gamma_1, gamma_2 = 5.31, 6.31

    target_Gamma_k = [Gamma_1, Gamma_2]
    target_omega_k = [omega_1, omega_2]
    target_gamma_k = [gamma_1, gamma_2]

    # array of time points
    time_grid = np.arange(.1,.5,.05).tolist() #make it a column vector

    # Instantiate model
    initial_Gamma_k = [1. * Gamma_1, Gamma_2]
    initial_omega_k =  [omega_1, omega_2]
    initial_gamma_k = [gamma_1, gamma_2]
    
    # Instantiate model
    model = PropagModel(time_grid=time_grid, initial_Gamma_k=initial_Gamma_k, initial_omega_k=initial_omega_k, initial_gamma_k=initial_gamma_k)
    
     
    
    # Before your model compilation, specify a custom learning rate
    custom_learning_rate = 0.01  # This is an example value; adjust based on your needs

    # Instantiate the Adam optimizer with the custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)

    # Compile the model with the custom optimizer
    model.compile(optimizer=optimizer, loss=custom_loss)

    x_dummy = tf.zeros((2,len(time_grid)), dtype=tf.float32)
  
    model_init = model(x_dummy)
    

    target_vector  = model.target_vector(tf.constant(target_Gamma_k, dtype=tf.float32), tf.constant(target_omega_k, dtype=tf.float32), tf.constant(target_gamma_k, dtype=tf.float32))
    
    record_parameters_callback = RecordParametersCallback()
    history = model.fit(x=x_dummy, y=target_vector, epochs=nbr_epochs, callbacks=[record_parameters_callback, PrintParametersCallback()])


    print('error_init', custom_loss(target_vector, model_init))
   
    print("initial propag", model_init)
    print("target propag", target_vector)   
    got = model(x_dummy)
    print("final propag", got)
    print("final error," , custom_loss(target_vector, got))
    print("trained parameters")
    print(model.Gamma_k)
    print(model.omega_k)
    print(model.gamma_k)

    #create a scatter plot where the dots are connected by a line
    plt.plot(np.arange(0,nbr_epochs,1),history.history['loss'][::1], 'o-')

    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.ylim(bottom=-0.1)  
    #add vertical line
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
  


    #plot the parameters over time
    gamma_k_values = record_parameters_callback.parameters_over_time['Gamma_k']
    plt.plot(gamma_k_values, label='Gamma_k')
    omega_k_values = record_parameters_callback.parameters_over_time['omega_k']
    plt.plot(omega_k_values, label='omega_k')
    gamma_k_values = record_parameters_callback.parameters_over_time['gamma_k']
    plt.plot(gamma_k_values, label='gamma_k')
    plt.title('Parameters values over Epochs')
    plt.legend()
    plt.xlabel('Epoch')
    #plt.axhline(y=np.pi, color='r', linestyle='--')
    #plt.axhline(y=2*np.pi, color='r', linestyle='--')
    plt.show()

    #plot propagator for every set of parameters
    propagator = []
    nbr_elements = len(gamma_k_values)
    for i in range(nbr_elements):
        propagator = propag_from_params(tf.reshape(tf.constant(time_grid, dtype=tf.float32),(-1,1)), tf.constant(gamma_k_values[i], dtype=tf.float32), tf.constant(omega_k_values[i], dtype=tf.float32), tf.constant(gamma_k_values[i], dtype=tf.float32)).numpy()
        plt.plot(np.arange(len(time_grid)), (propagator[0] - target_vector[0])/target_vector[0], color = 'blue', alpha = i / nbr_elements)
        plt.plot(np.arange(len(time_grid)), (propagator[1] - target_vector[1])/target_vector[0], color = 'red', alpha = i / nbr_elements)

    
    plt.show()
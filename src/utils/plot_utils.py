import numpy as np
from scipy.optimize import curve_fit


# Define the logarithmic function to fit
def log_func(x, a, b):
    return a * np.log(x) + b


def fit_logarithmic(x_data, y_data):
    """
    Fit a logarithmic curve to the given data points.

    :param x_data: array-like, independent data
    :param y_data: array-like, dependent data
    :return: parameters of the fitted logarithmic function
    """

    # Use curve_fit to find the best fit parameters
    params, _ = curve_fit(log_func, x_data, y_data)

    return params


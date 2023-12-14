import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

class LogarithmicCurveFitter:
    def __init__(self, x_data, y_data):
        """
        Initialize the LogarithmicCurveFitter with input data.

        Parameters:
            x_data (array-like): Independent variable data.
            y_data (array-like): Dependent variable data.
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.params = None

    def logarithmic_function(self, x, a, b):
        """
        Logarithmic function to fit the data.

        Parameters:
            x (array-like): Independent variable data.
            a (float): Coefficient for the logarithmic term.
            b (float): Coefficient for the constant term.

        Returns:
            array-like: Fitted values based on the logarithmic function.
        """
        return a * np.log(x) + b

    def fit_curve(self):
        """
        Fit the logarithmic curve to the input data using curve_fit.
        """
        self.params, covariance = curve_fit(self.logarithmic_function, self.x_data, self.y_data)

    def plot_data_and_fit(self, ax):
        """
        Plot the original data and the fitted logarithmic curve.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot.
        """
        if self.params is None:
            self.fit_curve()

        y_fit = self.logarithmic_function(self.x_data, *self.params)

        ax.scatter(self.x_data, self.y_data, label='Original Data')
        ax.plot(self.x_data, y_fit, label='Fitted Logarithmic Curve', color='red')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
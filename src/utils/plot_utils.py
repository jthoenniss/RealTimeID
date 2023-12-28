

def plot_data(ax, x_data, y_data, color, label):
    """Function to plot data on a given axis."""
    ax.plot(x_data, y_data, marker="o", linestyle="-", markersize=3, color=color, label=label)
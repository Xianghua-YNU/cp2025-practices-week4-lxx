"""
Least Squares Fitting and Photoelectric Effect Experiment
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    Load data from file.

    Parameters:
        filename: Path to the data file

    Returns:
        x: Array of x values
        y: Array of y values
    """
    try:
        data = np.loadtxt(filename)
        return data[:, 0], data[:, 1]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load file: {filename}") from e

def calculate_parameters(x, y):
    """
    Calculate least squares fitting parameters.

    Parameters:
        x: Array of x values
        y: Array of y values

    Returns:
        m: Slope of the fitted line
        c: Intercept of the fitted line
        Ex: Mean of x
        Ey: Mean of y
        Exx: Mean of x squared
        Exy: Mean of x*y
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input data cannot be empty")
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    
    N = len(x)
    Ex = np.mean(x)
    Ey = np.mean(y)
    Exx = np.mean(x**2)
    Exy = np.mean(x*y)
    
    denominator = Exx - Ex**2
    if denominator == 0:
        raise ValueError("Cannot calculate parameters: denominator is zero")
    
    m = (Exy - Ex*Ey) / denominator
    c = (Exx*Ey - Ex*Exy) / denominator
    
    return m, c, Ex, Ey, Exx, Exy

def plot_data_and_fit(x, y, m, c):
    """
    Plot data points and fitted line.

    Parameters:
        x: Array of x values
        y: Array of y values
        m: Slope of the fitted line
        c: Intercept of the fitted line

    Returns:
        fig: matplotlib figure object
    """
    if np.isnan(m) or np.isnan(c):
        raise ValueError("Slope and intercept cannot be NaN")
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Experimental Data')
    y_fit = m*x + c
    ax.plot(x, y_fit, 'r', label='Fitted Line')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    return fig

def calculate_planck_constant(m):
    """
    Calculate Planck's constant.

    Parameters:
        m: Slope of the fitted line

    Returns:
        h: Calculated Planck's constant
        relative_error: Relative error compared to the actual value
    """
    if m <= 0:
        raise ValueError("Slope must be positive")
    
    e = 1.602e-19  # Electron charge
    h = m * e
    actual_h = 6.626e-34
    relative_error = abs(h - actual_h) / actual_h * 100
    return h, relative_error

def main():
    """Main function"""
    try:
        # Data file path
        filename = "millikan.txt"
        
        # Load data
        x, y = load_data(filename)
        
        # Calculate fitting parameters
        m, c, Ex, Ey, Exx, Exy = calculate_parameters(x, y)
        
        # Print results
        print(f"Ex = {Ex:.6e}")
        print(f"Ey = {Ey:.6e}")
        print(f"Exx = {Exx:.6e}")
        print(f"Exy = {Exy:.6e}")
        print(f"Slope m = {m:.6e}")
        print(f"Intercept c = {c:.6e}")
        
        # Plot data and fitted line
        fig = plot_data_and_fit(x, y, m, c)
        
        # Calculate Planck's constant
        h, relative_error = calculate_planck_constant(m)
        print(f"Calculated Planck's constant h = {h:.6e} J·s")
        print(f"Relative error compared to actual value: {relative_error:.2f}%")
        
        # Save the figure
        fig.savefig("millikan_fit.png", dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

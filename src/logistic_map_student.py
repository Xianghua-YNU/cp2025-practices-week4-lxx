"""
Logistic Mapping and Chaos System Study
"""

import numpy as np
import matplotlib.pyplot as plt

def iterate_logistic(r, x0, n):
    """
    Iterate the Logistic map.

    Parameters:
        r: Growth rate parameter
        x0: Initial value
        n: Number of iterations

    Returns:
        x: Array of iterated values
    """
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def plot_time_series(r, x0, n):
    """
    Plot the time series of the Logistic map.

    Parameters:
        r: Growth rate parameter
        x0: Initial value
        n: Number of iterations

    Returns:
        fig: matplotlib figure object
    """
    x = iterate_logistic(r, x0, n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(n), x, 'b-', label=f'r = {r}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('x')
    ax.set_title(f'Logistic Map Time Series (r = {r})')
    ax.legend()
    ax.grid(True)
    return fig

def plot_bifurcation(r_min, r_max, n_r, n_iterations, n_discard):
    """
    Plot the bifurcation diagram of the Logistic map.

    Parameters:
        r_min: Minimum value of r
        r_max: Maximum value of r
        n_r: Number of r values
        n_iterations: Number of iterations for each r
        n_discard: Number of initial iterations to discard for each r

    Returns:
        fig: matplotlib figure object
    """
    r_values = np.linspace(r_min, r_max, n_r)
    x_values = []
    
    for r in r_values:
        x = 0.5  # Initial value
        # Discard the first n_discard iterations
        for _ in range(n_discard):
            x = r * x * (1 - x)
        # Record the next n_iterations - n_discard iterations
        for _ in range(n_iterations - n_discard):
            x = r * x * (1 - x)
            x_values.append((r, x))
    
    r_values_plot, x_values_plot = zip(*x_values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(r_values_plot, x_values_plot, s=0.1, c='k', marker='.')
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Logistic Map Bifurcation Diagram')
    ax.grid(True)
    return fig

def main():
    """Main function"""
    # Time series analysis
    r_values = [2.0, 3.2, 3.45, 3.6]
    x0 = 0.5
    n = 100
    
    for r in r_values:
        fig = plot_time_series(r, x0, n)
        fig.savefig(f"logistic_r{r}.png", dpi=300)
        plt.close(fig)
    
    # Bifurcation diagram analysis
    fig = plot_bifurcation(2.5, 4.0, 1000, 1000, 100)
    fig.savefig("bifurcation.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()

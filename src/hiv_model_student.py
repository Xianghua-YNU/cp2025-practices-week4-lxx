import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HIVModel:
    def __init__(self, A, alpha, B, beta):
        """
        Initialize model parameters.

        Parameters:
            A: Model parameter A
            alpha: Model parameter α
            B: Model parameter B
            beta: Model parameter β
        """
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta

    def viral_load(self, time):
        """
        Calculate viral load.

        Parameters:
            time: Time array

        Returns:
            Viral load array
        """
        return self.A * np.exp(-self.alpha * time) + self.B * np.exp(-self.beta * time)

    def plot_model(self, time, label=None):
        """
        Plot model curve.

        Parameters:
            time: Time array
            label: Curve label
        """
        viral_load = self.viral_load(time)
        plt.plot(time, viral_load, label=label)


def load_hiv_data(filepath):
    """
    Load HIV data.

    Parameters:
        filepath: Path to data file

    Returns:
        time_data: Time data array
        viral_load_data: Viral load data array
    """
    try:
        # Try loading .npz file
        hiv_data = np.load(filepath)
        time_data = hiv_data['time_in_days']
        viral_load_data = hiv_data['viral_load']
    except:
        # If .npz file doesn't exist, try loading .csv file
        hiv_data = np.loadtxt(filepath, delimiter=',')
        time_data = hiv_data[:, 0]
        viral_load_data = hiv_data[:, 1]
    return time_data, viral_load_data


def main():
    """
    Main function to test the model.
    """
    # Generate time series
    time = np.linspace(0, 10, 100)

    # Define different model parameters
    models = [
        HIVModel(A=1, alpha=1, B=0, beta=0),  # Only A and alpha are active
        HIVModel(A=1, alpha=2, B=0, beta=0),  # Increase alpha
        HIVModel(A=1, alpha=1, B=0.5, beta=0),  # Add B and beta
        HIVModel(A=1, alpha=1, B=0.5, beta=2),  # Increase beta
    ]

    # Plot model curves with different parameters
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        model.plot_model(time, label=f"Model {i+1}")
    plt.xlabel('Time (days)')
    plt.ylabel('Viral Load (V(t))')
    plt.title('HIV Viral Load Models')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/hiv_models.png', dpi=300)  # Save the figure
    plt.show()

    # Load experimental data
    time_data, viral_load_data = load_hiv_data('data/HIVseries.csv')  # Or 'HIVseries.npz'
    model = HIVModel(A=175000, alpha=0.6, B=0, beta=0)

    # Plot experimental data and model on the same figure
    plt.figure(figsize=(10, 6))
    plt.scatter(time_data, viral_load_data, color='blue', label='Experimental Data', marker='o')
    model.plot_model(time, label="Model")

    # Add labels and title
    plt.xlabel('Time (days)')
    plt.ylabel('Viral Load (V(t))')
    plt.title('HIV Viral Load Model Fitting')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/hiv_model_fitting.png', dpi=300)  # Save the figure
    plt.show()


if __name__ == "__main__":
    main()

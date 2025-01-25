import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_loss_with_epoch_and_time(losses, times, labels, data_name, num_initial_points):
    """
    Plots epoch-loss and cumulative time-loss graphs for each initial point and prints the values to the console.

    Parameters:
    - losses: A list of loss values for each optimizer per epoch
    - times: A list of time values for each optimizer per epoch
    - labels: A list of labels corresponding to each optimizer
    - data_name: The dataset name to use in the graph titles (e.g., "MNIST")
    - num_initial_points: The number of initial starting points
    """
     
    optimizer_colors = {
        "SGD": "r",
        "SGD with Momentum": "g",
        "Adam": "b"
    }

    for point_idx in range(num_initial_points):
        # Epoch-Loss 
        plt.figure(figsize=(10, 5))
        print(f"\nInitial Point {point_idx + 1} - Epoch vs Loss")
        for i, (loss, label) in enumerate(zip(losses, labels)):
            if f"Point {point_idx + 1}" in label:  # Check if the current optimizer corresponds to the current starting point
                optimizer_name = label.split(" - ")[0]
                color = optimizer_colors.get(optimizer_name, "k")  
                plt.plot(range(len(loss)), loss, label=f"{optimizer_name} Loss", color=color)

                for epoch, loss_value in enumerate(loss):
                    print(f"Optimizer: {optimizer_name}, Epoch: {epoch}, Loss: {loss_value:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{data_name} - Epoch vs Loss for Initial Point {point_idx + 1}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Cumulative Time-Loss 
        plt.figure(figsize=(10, 5))
        print(f"\nInitial Point {point_idx + 1} - Cumulative Time vs Loss")
        for i, (loss, time, label) in enumerate(zip(losses, times, labels)):
            if f"Point {point_idx + 1}" in label:  
                optimizer_name = label.split(" - ")[0]
                color = optimizer_colors.get(optimizer_name, "k") 
                
                
                cumulative_time = np.cumsum(time)  
                plt.plot(cumulative_time, loss, label=f"{optimizer_name} Time-Loss", color=color, linestyle="--")

                
                for time_value, loss_value in zip(cumulative_time, loss):
                    print(f"Optimizer: {optimizer_name}, Time: {time_value:.2f}, Loss: {loss_value:.4f}")
        plt.xlabel("Cumulative Time (seconds)")
        plt.ylabel("Loss")
        plt.title(f"{data_name} - Time vs Loss for Initial Point {point_idx + 1}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def visualize_trajectories(trajectories, labels, algorithms):
    """
    Visualizes the optimization trajectories in 2D space using t-SNE.

    Parameters:
    - trajectories: List of trajectory arrays for each optimizer and starting point
    - labels: List of labels corresponding to each trajectory
    - algorithms: List of algorithm names (e.g., ["SGD", "SGD with Momentum", "Adam"])
    """

    all_points = np.vstack(trajectories)
    tsne = TSNE(n_components=2, random_state=42, perplexity=27)
    points_2d = tsne.fit_transform(all_points)

    algorithm_colors = {
        "SGD": "r",
        "SGD with Momentum": "g",
        "Adam": "b"
    }

    plt.figure(figsize=(10, 8))
    start_idx = 0

    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        num_points = traj.shape[0]
        traj_2d = points_2d[start_idx : start_idx + num_points] # Transform the trajectory points to 2D
        start_idx += num_points

        algorithm_name = label.split(" - ")[0]
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], label=f"{algorithm_name}", 
                 color=algorithm_colors[algorithm_name], alpha=0.7)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Optimization Trajectories")
    plt.show()

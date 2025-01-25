import numpy as np
import torch
import time

device = torch.device("mps")

def initialize_weights(model):
    """
    Initializes the weights of the given model.

    Parameters:
    - model: PyTorch model whose weights need to be initialized.
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def reset_model_to_initial_weights(model, initial_weights):
    """
    Resets the model's weights to the given initial state.

    Parameters:
    - model: PyTorch model to reset.
    - initial_weights: State dictionary containing the initial weights.
    """
    model.load_state_dict(initial_weights)


def train_and_track_sst(model, iterator, optimizer, criterion, num_epochs):
    """
    Trains the model on the SST dataset and tracks parameter trajectories, losses, and times.

    Parameters:
    - model: PyTorch model to be trained.
    - iterator: Data iterator for the SST dataset.
    - optimizer: Optimizer to update model weights.
    - criterion: Loss function to compute the loss.
    - num_epochs: Number of epochs for training.

    Returns:
    - trajectory: Array containing parameter trajectories.
    - losses: List of average loss values for each epoch.
    - times: List of time durations for each epoch.
    """
    model.train()
    trajectory = []  # Tracks parameter trajectories
    losses = []  # Tracks loss values for each epoch
    times = []  # Tracks time duration for each epoch

    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time() 

        for batch in iterator:
            text, offsets = batch.text, None
            labels = batch.label.to(device)
            text = text.to(device)

            optimizer.zero_grad()
            predictions = model(text, offsets)
            loss = criterion(predictions, labels.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

       
        end_time = time.time()
        avg_epoch_loss = epoch_loss / len(iterator)
        losses.append(avg_epoch_loss)
        epoch_time = end_time - start_time
        times.append(epoch_time)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} - Time: {epoch_time:.2f} seconds")


        if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % 5 == 0:
            trajectory.append(model.fc.weight.data.clone().cpu().numpy().flatten())

    return np.array(trajectory), losses, times



def train_and_track_mnist(model, iterator, optimizer, criterion, num_epochs):
    """
    Trains the model on the MNIST dataset and tracks parameter trajectories, losses, and times.

    Parameters:
    - model: PyTorch model to be trained.
    - iterator: Data iterator for the MNIST dataset.
    - optimizer: Optimizer to update model weights.
    - criterion: Loss function to compute the loss.
    - num_epochs: Number of epochs for training.

    Returns:
    - trajectory: Array containing parameter trajectories.
    - losses: List of average loss values for each epoch.
    - times: List of time durations for each epoch.
    """
    model.train()
    trajectory = []
    losses = []
    times = []  

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        start_time = time.time()
        epoch_loss = 0

        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
       
        end_time = time.time()
        avg_epoch_loss = epoch_loss / len(iterator)
        losses.append(avg_epoch_loss)
        epoch_time = end_time - start_time
        times.append(epoch_time)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} - Time: {epoch_time:.2f} seconds")

        if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % 5 == 0:
            trajectory.append(model.fc3.weight.data.clone().cpu().numpy().flatten())

    return np.array(trajectory), losses, times


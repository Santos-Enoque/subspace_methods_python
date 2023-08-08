import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from src.models import KernelMSM
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


def plot_images(data):
    fig, axes = plt.subplots(5, 6, figsize=(15, 10))
    axes = axes.ravel()

    for i in range(30):  # For each class
        sample_image = data[i, 0, 0].reshape(24, 24)
        axes[i].imshow(sample_image, cmap='gray')
        axes[i].axis('off')  # Turn off the axis numbers
        axes[i].set_title(f"Class {i+1}")

    plt.tight_layout()
    plt.show()


def load_and_process_data(file_path):
    NUMBER_OF_SETS = 50
    
    # Load dataset
    data_dict = scipy.io.loadmat(file_path)
    data = data_dict["data"]

    # Ensure the data shape matches the expected shape
    assert data.shape == (24, 24, 60, 7, 30, 100)

    # Flatten image dimensions and split the data
    flattened_data = data.reshape(24*24, 60, 7, 30, 100)
    train_data = flattened_data[:, :, :, :, :80]
    test_data = flattened_data[:, :, :, :, 80:]

    # Reorder dimensions: classes, images/camera, cameras, participants, image_data
    train_data = train_data.transpose(3, 1, 2, 4, 0)
    test_data = test_data.transpose(3, 1, 2, 4, 0)

    # Combine samples into a single dimension
    train_data = train_data.reshape(30, -1, 576)
    test_data = test_data.reshape(30, NUMBER_OF_SETS, -1, 576)

    train_X = train_data
    train_y = np.arange(len(train_X))
    test_X = test_data
    test_X = test_X.reshape(-1, test_X.shape[-2], test_X.shape[-1])
    test_y = np.array([[i] * NUMBER_OF_SETS for i in range(30)]).flatten()
    
    return train_X, train_y, test_X, test_y


file_path = "data/TsukubaHandSize24x24.mat"
train_X, train_y, test_X, test_y = load_and_process_data(file_path)
print("Training data shape: {}".format(train_X.shape))
print("Training labels shape: {}".format(train_y.shape))
print("Test data shape: {}".format(test_X.shape))
print("Test labels shape: {}".format(test_y.shape))

print(train_y)

# Set the amount of random noise to add to the data
# noise_scale = 5e-1

# # Initialize a random number generator with a fixed seed
# rng = np.random.RandomState(seed=100)
# # Add random noise to the training data
# train_X = [_X + noise_scale * rng.randn(*_X[0].shape) for _X in train_X]

# # Add random noise to the test data
# test_X = [_X + noise_scale * rng.randn(*_X[0].shape) for _X in test_X]

# fit the model
model = KernelMSM(n_subdims=5, sigma=100, faster_mode=True)
model.fit(train_X, train_y)

pred = model.predict(test_X)
print(f"pred: {pred}\ntrue: {test_y}\naccuracy: {(pred == test_y).mean()}")
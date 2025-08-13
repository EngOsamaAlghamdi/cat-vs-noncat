from model_utils import model
import numpy as np
import h5py

# ✔ Load dataset
train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
test_dataset = h5py.File('data/test_catvnoncat.h5', "r")

train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])

# ✔ Preprocess
train_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255.
test_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255.
train_y = train_set_y_orig.reshape(1, -1)
test_y = test_set_y_orig.reshape(1, -1)

# ✔ Model structure (2-layer network)
layers_dims = [12288, 20, 1]

# ✔ Train
parameters = model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

# Optional: Evaluate test accuracy
from model_utils import forward_propagation
AL, _ = forward_propagation(test_x, parameters)
predictions = (AL > 0.5)
accuracy = np.mean(predictions == test_y)
print(f"Test accuracy: {accuracy * 100:.2f}%")

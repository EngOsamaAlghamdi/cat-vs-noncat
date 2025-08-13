import numpy as np

def initialize_parameters(layer_dims):
    """
    Initializes weights with small random values and biases with zeros.
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_propagation(X, parameters):
    """
    Implements forward propagation for a 2-layer network.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "W2": W2,
        "X": X
    }

    return A2, cache


def backward_propagation(AL, Y, cache):
    """
    Implements the backward propagation for a 2-layer network.
    """
    m = Y.shape[1]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = cache["W2"]

    dZ2 = AL - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # derivative of tanh
    dW1 = (1 / m) * np.dot(dZ1, cache["X"].T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.
    """
    L = len(parameters) // 2  

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


def compute_cost(AL, Y):
    """
    Computes the cross-entropy cost.
    """
    m = Y.shape[1]
    cost = - (1 / m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
    return np.squeeze(cost)


def train_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Trains a 2-layer neural network.
    """
    np.random.seed(1)
    parameters = initialize_parameters(layers_dims)
    costs = []

    for i in range(num_iterations):
        AL, cache = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)

    return parameters


# Wrapper so that main.py can call model()
def model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    return train_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost)

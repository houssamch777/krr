import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sign(x):
    return np.where(x < 0.5, -1, 1)
def sign_prime(x):
    return np.zeros_like(x)
def sigmoid_prime(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def step(x):
    return np.where(x >= 0, 1, -1)


def step_prime(x):
    return np.zeros_like(x)

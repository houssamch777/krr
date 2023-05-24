import numpy as np
import random
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.weights = np.array([[0.3], [-0.1]])
        print("Initial weights in the perceptron:", self.weights)
        print("****************************")

    def forward(self, X):
        net_input = np.dot(X, self.weights) - 0.2
        self.layer = np.vectorize(step)(net_input)
        return self.layer

    def backward(self, X, Y, output, learning_rate):
        error = Y - output
        delta = np.dot(X.T, error) * learning_rate
        self.weights += delta

    def train(self, X, Y, learning_rate, epochs):
        accuracies = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output, learning_rate)
            accuracy = self.accuracy(X, Y)
            accuracies.append(accuracy)
            print("Epoch:", epoch+1, "out of", epochs)
            print("Learning Rate:", learning_rate)
            print("Accuracy:", accuracy)
            print("Weights:", self.weights)
            print("**************************")
        return accuracies

    def accuracy(self, X, Y):
        predictions = self.forward(X)
        predicted_labels = np.where(predictions >= 0, 1, -1)
        accuracy = np.mean(predicted_labels == Y)
        return accuracy

def step(x):
    if x <= 0:
        return -1
    else:
        return 1

# Define the number of training and test samples
napp = 500
ntest = 500
x2 = np.concatenate([1.5*np.random.randn(2, napp//2) - 0.5, 0.7*np.random.randn(2, napp//2) + 1.5], axis=1)
y2 = np.concatenate([np.ones(napp//2), -np.ones(napp//2)])
xt2 = np.concatenate([1.5*np.random.randn(2, ntest//2) - 0.5, 0.7*np.random.randn(2, ntest//2) + 1.5], axis=1)
yt2 = np.concatenate([np.ones(ntest//2), -np.ones(ntest//2)])
# Dataset 1
x1 = x2
y1 = y2
xt1 = xt2
yt1 = yt2

# Training data
x_train = x1.reshape(napp, 2)
y_train = y1.reshape(napp, 1)

# Test data
x_test = xt1.reshape(ntest, 2)
y_test = yt1.reshape(ntest, 1)

# Learning rates to try
learning_rates = [0.1, 0.01, 0.001]

# Create and train perceptron for each learning rate
for learning_rate in learning_rates:
    perceptron = Perceptron()
    accuracies = perceptron.train(x_train, y_train, learning_rate, 100)

    # Plot accuracy during training
    plt.plot(accuracies, label="Learning Rate: " + str(learning_rate))

    print("Final Accuracy for Learning Rate", learning_rate, ":", accuracies[-1])
    print("==============================")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

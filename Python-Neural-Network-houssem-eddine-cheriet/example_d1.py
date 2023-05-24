import numpy as np
import matplotlib.pyplot as plt

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

napp = 500
ntest = 500

# Dataset 1
x1 = 2 * np.random.rand(2, napp) - 1
y1 = np.sign(x1[0])
xt1 = 2 * np.random.rand(2, ntest) - 1
yt1 = np.sign(xt1[0])

# Training data
x_train = np.array([x1[:, i].reshape(1, 2) for i in range(napp)])
y_train = np.array([y1[i].reshape(1, 1) for i in range(napp)])


def train_and_plot(learning_rate):
    # Network
    net = Network()
    net.add(FCLayer(2, 1))
    net.add(ActivationLayer(tanh, tanh_prime))

    # Train
    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs=10, learning_rate=learning_rate)

    xt1_reshaped = np.array([xt1[:, i].reshape(1, 2) for i in range(napp)])  # Reshape xt1 to (1, 2, ntest)
    yt1_reshaped = np.array([yt1[i].reshape(1, 1) for i in range(napp)])  # Reshape xt1 to (1, 2, ntest)

    # After training the network
    accuracy = net.accuracy(xt1_reshaped, yt1_reshaped)
    print('Accuracy for learning rate %.2f: %.2f' % (learning_rate, accuracy))

    # Test
    out = np.array(net.predict(xt1_reshaped))  # Convert to NumPy array

    # Plot the data points and the predicted class grid
    plt.figure()
    plt.scatter(xt1[0, :], xt1[1, :], c=out.flatten(), cmap='coolwarm', marker='x')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Predicted Class Grid (Learning Rate: %.2f)' % learning_rate)
    plt.colorbar()
    plt.show()
    return accuracy

# Train and plot for different learning rates
learning_rates = [0.1]
accuracy=[]
for lr in learning_rates:
    accuracy.append(train_and_plot(lr))
for lr in range(len(learning_rates)):
    print("Accuracy for learning rate",learning_rates[lr],accuracy[lr])
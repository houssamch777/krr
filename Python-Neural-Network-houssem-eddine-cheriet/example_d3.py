import numpy as np
import matplotlib.pyplot as plt

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh_prime,tanh
from losses import mse_prime,mse

napp = 500
ntest = 500
sigma = 0
nb = napp // 16

# Dataset 3
x3 = np.empty((2, 0))
y3 = np.empty((0,))
xt3 = np.empty((2, 0))
yt3 = np.empty((0,))

for i in range(-2, 2+1):
    for j in range(-2, 2+1):
        x3 = np.concatenate((x3, np.vstack((i + (1 + sigma) * np.random.rand(1, nb),
                                            j + (1 + sigma) * np.random.rand(1, nb)))), axis=1)
        y3 = np.concatenate((y3, (2 * np.remainder((i + j + 4), 2) - 1) * np.ones(nb)))

nb = ntest // 16

for i in range(-2, 2+1):
    for j in range(-2, 2+1):
        xt3 = np.concatenate((xt3, np.vstack((i + (1 + sigma) * np.random.rand(1, nb),
                                              j + (1 + sigma) * np.random.rand(1, nb)))), axis=1)
        yt3 = np.concatenate((yt3, (2 * np.remainder((i + j + 4), 2) - 1) * np.ones(nb)))

x_train = np.array([x3[:, i].reshape(1, 2) for i in range(napp)])
y_train = np.array([y3[i].reshape(1, 1) for i in range(napp)])

# Network
net = Network()
net.add(FCLayer(2, 32))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(32, 16))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(16, 1))
net.add(ActivationLayer(tanh, tanh_prime))
# Train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=5000, learning_rate=0.01)
# Test
xt3_reshaped = np.array([xt3[:, i].reshape(1, 2) for i in range(ntest)])  # Reshape xt3 to (1, 2, ntest)
yt3_reshaped = np.array([yt3[i].reshape(1, 1) for i in range(napp)])  # Reshape xt1 to (1, 2, ntest)
# After training the network
accuracy = net.accuracy(xt3_reshaped, yt3_reshaped)
print('Accuracy: %.2f' % accuracy)
out = np.array(net.predict(xt3_reshaped)).flatten()  # Flatten the output array
# Use only the first ntest samples for plotting
xt3_plot = xt3[:, :ntest]
yt3_plot = yt3[:ntest]
out_plot = out[:ntest]
plt.figure(figsize=(8, 6))
plt.scatter(xt3_plot[0, :], xt3_plot[1, :], c=yt3_plot.flatten(), cmap='coolwarm', alpha=0.7)
plt.scatter(xt3_plot[0, :], xt3_plot[1, :], c=out_plot, cmap='coolwarm', edgecolor='black', s=40)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Predicted Class Grid')
cbar = plt.colorbar()
cbar.set_label('Predicted Class')
# Plot decision boundary
h = 0.02  # Step size for the mesh grid
x_min, x_max = xt3_plot[0, :].min() - 1, xt3_plot[0, :].max() + 1
y_min, y_max = xt3_plot[1, :].min() - 1, xt3_plot[1, :].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten the mesh grid points
Z = np.array(net.predict(grid_points)).flatten()  # Flatten the predicted classes
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='dashed')

plt.tight_layout()
plt.show()
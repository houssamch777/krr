import numpy as np
#import matplotlib.pyplot as plt

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
# Dataset 2
x2 = np.concatenate((1.5 * np.random.randn(2, round(napp/2)) - 0.5,
                     0.7 * np.random.randn(2, round(napp/2)) + 1.5), axis=1)
y2 = np.concatenate((np.ones(round(napp/2)), -np.ones(round(napp/2))))
xt2 = np.concatenate((1.5 * np.random.randn(2, round(ntest/2)) - 0.5,
                      0.7 * np.random.randn(2, round(ntest/2)) + 1.5), axis=1)
yt2 = np.concatenate((np.ones(round(ntest/2)), -np.ones(round(ntest/2))))
# training data
x_train = np.array([x2[:, i].reshape(1, 2) for i in range(napp)])
y_train =  np.array([y2[i].reshape(1, 1) for i in range(napp)])


# network
net = Network()
net.add(FCLayer(2, 4))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(4, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10, learning_rate=0.1)


import matplotlib.pyplot as plt
# Test
xt2_reshaped = np.array([xt2[:, i].reshape(1, 2) for i in range(napp)])  # Reshape xt1 to (1, 2, ntest)
yt2_reshaped = np.array([yt2[i].reshape(1, 1) for i in range(napp)])  # Reshape xt1 to (1, 2, ntest)
# After training the network
accuracy = net.accuracy(xt2_reshaped, yt2_reshaped)
print('Accuracy: %.2f' % accuracy)





out = np.array(net.predict(xt2_reshaped))  # Convert to NumPy array


# Plot the data points and the predicted class grid
plt.figure(figsize=(8, 6))
#plt.scatter(xt2[0, :], xt2[1, :], c=yt2.flatten(), cmap='coolwarm', alpha=0.7, s=40)
plt.scatter(xt2[0, :], xt2[1, :], c=out.flatten(), cmap='coolwarm', marker='x', s=40)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Predicted Class Grid')
cbar = plt.colorbar()
cbar.set_label('Predicted Class')
plt.tight_layout()
plt.show()

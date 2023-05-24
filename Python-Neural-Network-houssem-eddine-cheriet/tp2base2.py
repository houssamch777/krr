import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime,sigmoid,sigmoid_prime,sign,sign_prime
from losses import mse, mse_prime

np.random.seed(42)  # for reproducibility

napp = 500
ntest = 500
sigma = 0  # increase to mix at the boundaries
nb = napp // 16
x3, y3 = [], []
xt3, yt3 = [], []
for i in range(-2, 2 + 1):
    for j in range(-2, 2 + 1):
        x3.append([i + (1 + sigma) * np.random.rand(nb), j + (1 + sigma) * np.random.rand(nb)])
        y3.append((2 * ((i + j + 4) % 2) - 1) * np.ones(nb))

x3, y3 = np.concatenate(x3, axis=1), np.concatenate(y3)
nb = ntest // 16
for i in range(-2, 2 + 1):
    for j in range(-2, 2 + 1):
        xt3.append([i + (1 + sigma) * np.random.rand(nb), j + (1 + sigma) * np.random.rand(nb)])
        yt3.append((2 * ((i + j + 4) % 2) - 1) * np.ones(nb))
xt3, yt3 = np.concatenate(xt3, axis=1), np.concatenate(yt3)

# Reshape x3 and y3
x3 = x3.reshape(-1, 1, 2)
y3 = y3.reshape(-1, 1, 1)
xt3 = xt3.reshape(-1, 1, 2)
yt3 = yt3.reshape(-1, 1, 1)

# network
net = Network()
net.add(FCLayer(2, 16))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(16, 1))
net.add(ActivationLayer(sign, sign_prime))


# train
net.use(mse, mse_prime)
net.fit(x3, y3, epochs=1000, learning_rate=0.1)

# test
out = net.predict(xt3)
print(out)
accuracy = np.mean(np.round(out) == yt3)
print("Accuracy: {:.2%}".format(accuracy))
out = np.array(out)
# Reshape the predictions for plotting
out = out.reshape(-1)
import matplotlib.pyplot as plt
# Plot the points
def sign(x):
    if x>0.5:
        return 1
    return -1
out_sign = np.vectorize(sign)(out)
plt.scatter(xt3[:, 0, 0], xt3[:, 0, 1], c=out_sign)
plt.colorbar()

# Set plot labels
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted Output')

# Show the plot
plt.show()
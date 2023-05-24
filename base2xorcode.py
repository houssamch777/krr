import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        self.weights = [[np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)],
                        [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)],
                        [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)]]
        self.biases = [np.random.uniform(-0.5, 0.5),
                       np.random.uniform(-0.5, 0.5),
                       np.random.uniform(-0.5, 0.5)]
        print("Initial weights and biases:")
        print("weights:", self.weights)
        print("biases:", self.biases)
        print("===========================")

    def forward(self, x, aff=True):
        # Hidden layer
        hidden_input3 = self.weights[0][0] * x[0] + self.weights[1][0] * x[1] + self.biases[0]
        hidden_output3 = sigmoid(hidden_input3)
        # Output layer
        hidden_input4 = self.weights[0][1] * x[0] + self.weights[1][1] * x[1] + self.biases[1]
        hidden_output4 = sigmoid(hidden_input4)
        # Output layer
        output_input = self.weights[2][0] * hidden_output3 + self.weights[2][1] * hidden_output4 + self.biases[2]
        output = sigmoid(output_input)
        if not aff:
            return output, hidden_input3, hidden_input4
        return output

    def backward(self, epochs, x, y, output, hidden_output1, hidden_output2, learning_rate=0.8, aff=True):
        # Output layer
        output_error = y - output
        output_delta5 = output_error * sigmoid_derivative(output)
        self.biases[2] += learning_rate * output_delta5
        self.weights[2][0] += hidden_output1 * learning_rate * output_delta5
        self.weights[2][1] += hidden_output2 * learning_rate * output_delta5
        hidden_delta3 = self.weights[2][0] * output_delta5 * sigmoid_derivative(hidden_output1)
        self.biases[0] += learning_rate * hidden_delta3
        self.weights[0][0] += x[0] * learning_rate * hidden_delta3
        self.weights[0][1] += x[1] * learning_rate * hidden_delta3
        hidden_delta4 = self.weights[2][1] * output_delta5 * sigmoid_derivative(hidden_output2)
        self.biases[1] += learning_rate * hidden_delta4
        self.weights[1][0] += x[0] * learning_rate * hidden_delta4
        self.weights[1][1] += x[1] * learning_rate * hidden_delta4
        if not aff and epochs % 10000 == 0:
            print("| x1 =", x[0], "| x2 =", x[1], "| y =", y, " | 𝜕5 =", output_delta5, "| 𝜷5 =", self.biases[2],
                  "| w35 =", self.weights[2][0], "| w45 =", self.weights[2][1], "| 𝜕3 =", hidden_delta3, "| 𝜷3 =",
                  self.biases[0],
                  "| w13 =", self.weights[0][0], "| w14 =", self.weights[0][1], "| 𝜕4 =", hidden_delta4, "| 𝜷4 =",
                  self.biases[1],
                  "| w23 =", self.weights[1][0], "| w24 =", self.weights[1][1], "| yc =", output, "| y3 =",
                  sigmoid(hidden_output1),
                  "| y4 =", sigmoid(hidden_output2))

    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        predictions = []
        accuracy = []
        for epoch in range(epochs):
            predictions = []
            if epoch % 10000 == 0:
                print("===================================================================================================================")
                print("Epoch:", epoch)

            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                output, hidden_output1, hidden_output2 = self.forward(x, aff=False)
                self.backward(epoch, x, y, output, hidden_output1, hidden_output2, learning_rate, aff=False)
                predictions.append(output)
            # Calculate accuracy
            predicted_labels = np.where(np.array(predictions) >= 0.5, 1, -1)
            acc = np.mean(predicted_labels == Y)
            accuracy.append(acc)

        return accuracy, predictions


# Create training data
napp = 100
ntest = 100
x2 = np.concatenate([1.5 * np.random.randn(2, napp // 2) - 0.5, 0.7 * np.random.randn(2, napp // 2) + 1.5], axis=1)
y2 = np.concatenate([np.ones(napp // 2), -np.ones(napp // 2)])

xt2 = np.concatenate([1.5 * np.random.randn(2, ntest // 2) - 0.5, 0.7 * np.random.randn(2, ntest // 2) + 1.5], axis=1)
yt2 = np.concatenate([np.ones(ntest // 2), -np.ones(ntest // 2)])

learning_rates = [0.01, 0.1, 0.8]
accuracy_plots = []
prediction_plots = []

# Create and train neural network for each learning rate
for lr in learning_rates:
    nn = NeuralNetwork()
    accuracy, predictions = nn.train(x2.T, y2, epochs=80, learning_rate=lr)
    accuracy_plots.append(accuracy)
    prediction_plots.append(predictions)

# Plotting the accuracy for different learning rates
fig, ax = plt.subplots()
for i, lr in enumerate(learning_rates):
    ax.plot(range(len(accuracy_plots[i])), accuracy_plots[i], label='Learning Rate: {}'.format(lr))

ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy over Epochs for Different Learning Rates")
ax.legend()
plt.show()

# Plotting the predictions for different learning rates
fig, axs = plt.subplots(len(learning_rates), sharex=True, sharey=True, figsize=(8, 6))

for i, lr in enumerate(learning_rates):
    axs[i].scatter(xt2.T[:, 0], xt2.T[:, 1], c=prediction_plots[i], cmap='viridis')
    axs[i].set_title("Predictions for Learning Rate: {}".format(lr))

axs[-1].set_xlabel("Input X1")
axs[-1].set_ylabel("Input X2")
plt.tight_layout()
plt.show()

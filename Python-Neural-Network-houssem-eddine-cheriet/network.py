import matplotlib.pyplot as plt
import numpy as np
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):

        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    def accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        correct = 0
        total = len(y_test)

        for i in range(total):
            predicted_class = round(predictions[i][0][0])
            if predicted_class == y_test[i][0][0]:
                correct += 1

        accuracy = correct / total
        return accuracy
    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # Lists to store accuracy and loss values
        accuracies = []
        losses = []

        # training loop
        for i in range(epochs):
            err = 0
            correct_count = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                il=0
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    #print("output of layer",il,": ",output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)


                # Calculate accuracy
                predicted = np.round(output)
                if predicted == y_train[j]:
                    correct_count += 1

            # calculate average error and accuracy on all samples
            err /= samples
            accuracy = correct_count / samples

            # Append accuracy and loss values to the lists
            accuracies.append(accuracy)
            losses.append(err)

            print('Epoch %d/%d   Error: %f   Accuracy: %.2f%%' % (i + 1, epochs, err, accuracy * 100))

        # Plot accuracy and loss
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), accuracies, label='Accuracy')
        plt.plot(range(1, epochs + 1), losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Accuracy and Loss During Training')
        plt.legend()
        plt.grid(True)
        plt.show()

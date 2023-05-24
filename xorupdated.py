import math
import numpy as np

def loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def accuracy(nn, X, y_true):
    correct = 0
    for i in range(len(X)):
        y_pred = nn.forward(X[i])
        if y_pred > 0.5 and y_true[i] == 1:
            correct += 1
        elif y_pred <= 0.5 and y_true[i] == 0:
            correct += 1
    return (correct / len(X)) * 100

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self):
                        #w13  w14    w23  w24    w35   w45
        self.weights = [[0.5, 0.9], [0.4, 1.0], [-1.2, 1.1]]
                    #  0-3   0-4  0-5
        self.biases = [0.8, -0.1, 0.3]
        self.errors=[]
        self.acc=[]
        self.losval=[]

        print("Initial weights and biases:")
        print("weights:", self.weights)
        print("biases:", self.biases)
        print("===========================")

    def forward(self, x,aff=True):
        # Hidden layer
        hidden_input3 = self.weights[0][0] * x[0] + self.weights[1][0] * x[1] + self.biases[0]
        hidden_output3 = sigmoid(hidden_input3)
        # Output layer
        hidden_input4 = self.weights[0][1] * x[0] + self.weights[1][1] * x[1]+ self.biases[1]
        hidden_output4 = sigmoid(hidden_input4)
        # Output layer
        output_input = self.weights[2][0] *hidden_output3 + self.weights[2][1] * hidden_output4 +self.biases[2]
        output = sigmoid(output_input)
        if not aff:
            return output,hidden_input3,hidden_input4
        return output

    def backward(self,epochs, x, y,output,hidden_output1,hidden_output2,learning_rate=0.1,aff=True):
        # Output layer
        output_error = y - output #e
        self.losval.append([loss(y,output_error),epochs])
        self.errors.append([output_error,epochs])
        output_delta5 = output_error * sigmoid_derivative(output) #sigma5=sig'(outpet)*e
        self.biases[2]+=learning_rate*output_delta5 #O-5
        self.weights[2][0]+=hidden_output1*learning_rate*output_delta5        #w35
        self.weights[2][1]+= hidden_output2 * learning_rate * output_delta5   #w45
        hidden_delta3=self.weights[2][0]*output_delta5*sigmoid_derivative(hidden_output1) #sigma 3
        self.biases[0]+=learning_rate*hidden_delta3 #O-3
        self.weights[0][0]+=x[0]*learning_rate*hidden_delta3 #w13
        self.weights[0][1]+=x[1]* learning_rate * hidden_delta3 #w14
        hidden_delta4=self.weights[2][1]*output_delta5*sigmoid_derivative(hidden_output2) #sigma 4
        self.biases[1]+=learning_rate*hidden_delta4 #O-4
        self.weights[1][0]+=x[0]*learning_rate*hidden_delta4 #w23
        self.weights[1][1]+=x[1] * learning_rate * hidden_delta4 #w24
        if not aff and epochs%10000==0:
            print("| x1 = ",x[0],"| x2 = ",x[1],"| y = ",y," | \u03C35 = ",output_delta5,"| \u03b85 = ",self.biases[2],"| w35 = ",self.weights[2][0],"| w45 = ",self.weights[2][1],"| \u03C33 = ",hidden_delta3,"| \u03b83 = ",self.biases[0],"| w13 = ", \
            self.weights[0][0],"| w14 = ",self.weights[0][1],"| \u03C34 = ",hidden_delta4,"| \u03b84 = ",self.biases[1],"| w23 = ",self.weights[1][0],"| w24 = ",self.weights[1][1],"| yc = ",output,"| y3 = ",sigmoid(hidden_output1),"| y4 = ",sigmoid(hidden_output2))


    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            if  epoch%10000==0:
                print("==================================================================================================================================\
                             ================================================================================================================================")
                print("Epoch:", epoch)
            self.acc.append([accuracy(self, X, Y), epoch])
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                output,hidden_output1,hidden_output2 = self.forward(x,aff=False)
                self.backward(epoch,x, y, output,hidden_output1,hidden_output2,aff=False)


# Create training data
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Create and train neural network
nn = NeuralNetwork()
nn.train(x,y,100000)
print("=========================================================================================================================================================")

print("Inputs and Predictions  for Xor function :")
for i in range(len(x)):
    if nn.forward(x[i])>0.5:
        print(x[i], "-> 1 ~~", nn.forward(x[i]))
    else:
        print(x[i], "-> 0 ~~", nn.forward(x[i]))

import matplotlib.pyplot as plt

predictions = [nn.forward(i) for i in x]
plt.scatter([i[0] for i in x], [i[1] for i in x], c=predictions)
plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.title("Predictions and Inputs for Xor function")
plt.show()


# Calculate accuracy
acc = accuracy(nn, x, y)
print("Accuracy: {:.2f}%".format(acc))

import numpy as np
import matplotlib.pyplot as plt

# Create a meshgrid to cover the range of input values
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X = np.column_stack((xx.ravel(), yy.ravel()))

# Use the neural network to predict output values for each point on the meshgrid
predictions = [nn.forward(i) for i in X]
predictions = np.array(predictions).reshape(xx.shape)

# Plot the meshgrid and the decision boundary
plt.contour(xx, yy, predictions, levels=[0.5], cmap="RdBu")
plt.scatter([i[0] for i in x], [i[1] for i in x], c=y, cmap="RdBu")
plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.show()
print(len(nn.errors))
print(nn.errors)

plt.plot([e[1] for e in nn.errors], [e[0] for e in nn.errors])
plt.show()
plt.plot([e[1] for e in nn.losval], [e[0] for e in nn.losval])
plt.show()
plt.plot([e[1] for e in nn.acc], [e[0] for e in nn.acc])
plt.show()

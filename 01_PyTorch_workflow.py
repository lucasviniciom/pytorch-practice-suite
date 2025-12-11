## 01. PyTorch Workflow

# Data, prepare and load
# Build model
# Fitting model/training
# Making predictions/inference
# Saving and loading model

import torch 
from torch import nn 
# contains all of PyTorch building blocks for neural networks
import numpy as np
import matplotlib.pyplot as plt
#Check PyTorch version
print(torch.__version__)

# Data, prepare and load
 
# ML can be in resume:
# Get data into numerial representation
# Build a model to learn patterns in that representation

# Example - Using linear regression to make line with known parameters

# Create known parameters
weight = 0.7
bias = 0.3

# Create dataset
start = 0 
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10], len(X), len(y))

#As we have an input and output, we then want to discover their relationship with NN
#

#Split in train and test sets - Validation set not always used
# Train 80%, Test 20%
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train),len(y_train),len(X_test),len(y_test))

#Visualize data
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    plt.figure(figsize=(10,7))
    #Plot training data
    plt.scatter(train_data, train_labels, c="b",s=4, label="Training Data")
    #Plot test data
    plt.scatter(test_data, test_labels, c="g",s=10, label="Testing Data")
    
    if predictions is not None:
        #Plot test data
        plt.scatter(test_data, predictions, c="o",s=4, label="Predictions")
    #show lagend
    plt.legend(prop={"size": 14})
    plt.show()

#plot_predictions()

# First PyTorch model
# Linear regression model class

##
#What the model does
# Starts with random values - weight and bias
# Look at the training data and adjusts the random values to better represent, or get closer
# to the ideal values - the weight and values we created the data with
##
#How it do so? With two algorithms:
# Gradient descent
# Backpropagation
class LinearRegression(nn.module): #almost everything in PyTorch inherits from this 
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        # Forward method to define the computation in the model
        def forward(self, x:torch.Tensor) -> torch.Tensor: # "x" is the input data
            return self.weights * x + self.bias # equivalent to linear regression formula

## PyTorch model building essentials
# torch.nn - contains all building blocks for graphs(another word for NN)
# torch.nn.Parameter - what parameter should our model try and learn
# torch.nn.Module - the base class for all NN modules
# if we subclass it, we need to overwrite forward()
# torch.optim - this is where otimizers in PyTorch live, they will help with gradient descent
# def forward() - All nn.Module subclasses require to overwrite
# it defines what happens in the forward computation


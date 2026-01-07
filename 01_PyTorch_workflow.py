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
import time
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
        plt.scatter(test_data, predictions, c="r",s=4, label="Predictions")
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
class LinearRegression(nn.Module): 
    #almost everything in PyTorch inherits from this 
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

# Checking the contents of PyTorch model
# To see what's inside, we can check out using parameters()

torch.manual_seed(42)

# Create instance of the model
model_0 = LinearRegression()

print(model_0)
print(model_0.parameters())
print(list(model_0.parameters()))
print((model_0.state_dict()))

## Test how well the random initial values makes a prediction
## When we pass data through our model, it's going to run through forward() method
# Make predictions with model
# Using this mode doesn't keep track of all grad desc calculations, so it runs faster
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)
print(y_test)

# plot_predictions(predictions=y_preds)

## Train Model
# Move from unknown parameters(these might be random), to a known, better representation

# To measure how poor/wrong the predictions are, we use a loss function
# Loss Function might be called Cost Function or criterion. 
#
# Optimizer - Takes into account loss of model and adjust parameters
# (in this case weight and bias) to improve the loss function
# Inside the optimizer, generally we set
# params - the model parameters we want to optimize
# lr - the learning rate, a hyperparam that defines how big/small the optimizer changes
# the parameters with each step

# Specifically in Pytorch we need:
# Training loop and testing loop

# Setup loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (choosed stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

## Building training and testing loop 
# We need in a training loop:
# 0 Loop through the data
# 1 Forward pass - involves data moving through our model's forward() functions
# also called forward propagation
# 2 Calculate the loss, comparing forward pass predictions to ground truth labels
# 3 Optimizer zero grad
# 4 Loss backwards - move backwards through the network to calculate the gradients
# of each of the parameters of the model, with respect to the loss (**backpropagation**)
# 5 Optimizer step - use the optimizer to adjust our models parameters to try 
# and improve the loss (**gradient descent**)

print((model_0.state_dict()))
torch.manual_seed(42)
# Epoch is one loop through the data
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

start = time.time()
for epoch in range(epochs):
    # Set model to training modeW
    # Train mode sets all parameters that require gradients to require gradients 
    model_0.train()

    # 1 Forward pass
    y_pred = model_0(X_train)

    # 2 Calculate the loss
    loss = loss_fn(y_pred, y_train)
    #print(loss)
    # 3 Optimizer zero grad
    optimizer.zero_grad()

    # 4 Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5 Step the optimizer (perform gradient descent)
    # By default the optimizer will accumulate through the loop, that's why we zero above
    optimizer.step()

    ####Testing 
    #turns off gradient descent - different setting we don't need for testing
    model_0.eval()

    with torch.inference_mode():
        # 1 Forward pass
        test_pred = model_0(X_test)

        # 2 Calculate loss
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch:{epoch}, Loss:{loss}, Test loss: {test_loss}")


print((model_0.state_dict()))

print(" time:", time.time() - start)
#print(loss)
#with torch.inference_mode():
#    y_preds_new = model_0(X_test)
#print(y_preds_new)
#print(y_test)
#
#plot_predictions(predictions=y_preds_new)

plt.plot(epoch_count,torch.tensor(loss_values),label="Train loss")
plt.plot(epoch_count,torch.tensor(test_loss_values),label="Test loss")
plt.title("Training and loss curves")
plt.xlabel('Loss')
plt.ylabel('Epochs')
#plt.show()

## Saving a model
# torch.save() - save Pytorch object in pickle format
# toarch.load() - allows to load Pytorch object
# torch.nn.Module.load_state_dict() - loads a model's saved state dictionary

from pathlib import Path

# Directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
 
# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH/ MODEL_NAME
print(MODEL_SAVE_PATH)

#Save model state dict
print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

# Load model
# Since we save the state_dict rather than the complete model
# we'll create a new instance of our model class and load it 

loaded_model_0 = LinearRegression()
#Loaded a new instance, with random values
#print(loaded_model_0.state_dict())

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())

print('Putting it all together')
print('------ Model 1----')
## --------------
## Putting it all together
##
import torch
from torch import nn
import matplotlib.pyplot as plt
# Create device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create data as y = weight*X + bias
weight = 0.7
bias = 0.3

start = 0
end = 1 
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Plot data
#plot_predictions(X_train,y_train,X_test,y_test)

# Building PyTorch Linear Model
# Different approach than before
# This one initializes the weight and bias automatically
# Also, for the forward method
# it will perform the computations pre-defined in nn.Linear
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        ## Use nn.Linear() for creating model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1, model_1.state_dict())

#Training

# Loss function
loss_fn = nn.L1Loss() 

# Optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

# Training loop
torch.manual_seed(42)

epochs = 200

for epoch in range(epochs):
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred,y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Optimizer step 
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred,y_test)

    # Print out whats happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss:{loss} | Test loss: {test_loss}")

print(model_1.state_dict())

# Making and evaluating predictions

# Turn model in evaluation mode
model_1.eval()

# Make predictions on the test data.
with torch.inference_mode():
    y_preds = model_1(X_test)
print(y_preds)

#Check predictions visually
#plot_predictions(predictions=y_preds)

#Saving model
from pathlib import Path

# Directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
 
# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH/ MODEL_NAME
print(MODEL_SAVE_PATH)

#Save model state dict
print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

# Load model
loaded_model_1 = LinearRegressionModelV2()
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_1.to(device)
print(loaded_model_1.state_dict())
print(next(loaded_model_1.parameters()).device)
# Eval model
loaded_model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)
print(y_preds)


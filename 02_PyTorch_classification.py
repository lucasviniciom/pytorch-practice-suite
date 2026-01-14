## Neural network classification with PyTorch

import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

# Make 1000 samples
n_samples = 1000
# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
print(f"First 5 samples of X: {X[:5]}")
print(f"First 5 samples of y: {y[:5]}")

circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:,1],
                        "label": y})
print(circles.head(10))

plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)
#plt.show()

## Check input and output shapes
print(X.shape, y.shape)

## Turn data in tensors and create train and test splits
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#print(X[:5],y[:5])

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Building model
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Subclass nn.Module 
# 2. Create 2 nn.Linear layers that are capable of handling the shapes of our data
# 3. Define forward() method that outlines the forward pass
# 4. Instantiate an instance of the model class and send to target device

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=5)
        self.layer_2 = nn.Linear(in_features=5,
                                 out_features=1)
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output

model_0 = CircleModelV0().to(device)
print(model_0)
print(next(model_0.parameters()).device)

# Replica model above using nn.Sequential()

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1),
).to(device)

print(model_0)
print(next(model_0.parameters()).device)
print(model_0.state_dict())

# Make predictions without training, just to check values
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Lenght of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Lenght of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 labels:\n{y_test[:10]}")

## 2.1 Setup loss function and optimizer
# This is problem specific
# For classification, in this case
# we want binary cross entropy or categorical cross entropy
#
# As reminder, loss functions measures how wrong the predictions are

# This one includes sigmoid activation function
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Calculate accuracy 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

## Train model



# Going from raw logits -> prediction probabilities -> prediction labels
# Our model outputs are going to be raw logits
# we can convert them into prediction probabilities with a activation functions
# e.g. sigmoid for binary classification, and softmax for multiclass
#
# Then we can convert our model's prediction probabilities to prediction labels
# either by rounding or taking the argmax()

# View first outputs of the forward pass on the test data
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
# Turn logits into prediction probabilities with sigmoid activation
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
# For our probabilities values, we need to perform a range-style rounding 
# if >= 0.5, then is class 1
# if < 0.5, then is class 0 
y_preds = torch.round(y_pred_probs)
print(y_preds)

# In full, in one line (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
# Check equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

###
# Building training and test loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

#Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # Calculate loss/accuracy
    ## nn.BCEWithLogitsLoss expects raw logits as inputs
    loss = loss_fn(y_logits, 
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward - backpropagation
    loss.backward()

    # Optimizer step - gradient descent
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # calculate test loss
        test_loss = loss_fn(test_logits, 
                   y_test)
        test_acc = accuracy_fn(y_true=y_test,
                      y_pred=test_pred)

        # Print
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss:{loss:.5f} | Acc: {acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")

# Make predictions and evaluate the model
# First version looks like the model isn't learning anything
# Let's inspect why 

import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
    print("help_functions.py already exists, skipping download")
else:
    print("Download helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundary of the model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

plt.show()


# Improving a model
# Add more layers - give the model more chances to learn patterns
# Add more hidden units
# Fit for longer
# Changing the activation functions
# Change the learning rate
# Change the loss function

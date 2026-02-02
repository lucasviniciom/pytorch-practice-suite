
# Putting it all together with multi-class classification problem
# Binary classification - One thing or another
# Multiclass classification - More than 2 classes

# Creating a toy multi-class dataset

import sklearn
from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary

# Set the hyperparameters for data creating
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create multiclass data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

device = "cuda" if torch.cuda.is_available else "cpu"


# Calculate accuracy 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

# Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)
#print(X_blob[:10])
#print(torch.unique(y_blob))
#We have 2 input features and 4 classes,outputs

#print(y_blob)
# Plot data
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0],X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu )

#plt.show()

X_blob_train, X_blob_test= X_blob_train.to(device), X_blob_test.to(device)
y_blob_train, y_blob_test= y_blob_train.to(device), y_blob_test.to(device)

## Building multi-class classification model 


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multi-class classification model
        
        Args:
            input_features(int) : Number of input features to model
            output_features(int) : Number of output features to model - classes
            hidden_units (init) : number of hidden units between layers, default 8
        
        Returns:

        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)
    
# Create an instance of BlobModel and sent to device
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

print(model_4)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

# Test without training
# We have as output 4 logits
# We need to convert them to prediction probabilities and the to prediction labels

#model_4.eval()
#with torch.inference_mode():
#    y_logits = model_4(X_blob_test)
#y_pred_probs = torch.softmax(y_logits, dim=1)
# Softmax evaluates each sample, in this case 4 numbers, and shows 
# The most likely to be the label
# We need to take it then with argmax. The highest value is the one predicted
#
# Logits (raw output of model) ->
# Pred probabilities (using torch.softmax)
# Pred label (taking argmax of pred probabilities)
#y_preds = torch.argmax(y_pred_probs, dim=1)
#print(y_logits[:10])
#print(y_pred_probs[:10])
#print(y_preds[:10])
#print(y_blob_test[:10])

# Training loop
epochs = 100
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in range(epochs):
    model_4.train()
    
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)

        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_preds)
        
    # Print out
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc:{test_acc:.2f}%")


# Predictions:
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
    y_pred_probs = torch.softmax(y_logits, dim=1).argmax(dim=1)
print(y_pred_probs[:10], y_blob_test[:10])

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)

plt.show()

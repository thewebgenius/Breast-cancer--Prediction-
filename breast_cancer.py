import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
#cell 2
data = load_breast_cancer(return_X_y = False, as_frame = True)
X = data.data
y = data.target
print(X.shape)
print(y.shape)
print(data.target.head())
#cell 3
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
#cell4 
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)
#cell 5
# ✅ Use scaled data here (this was your main issue)
X_train_torch = torch.from_numpy(X_train_scaled)
X_test_torch = torch.from_numpy(X_test_scaled)
y_train_torch = torch.from_numpy(y_train.to_numpy()).view(-1, 1)
y_test_torch = torch.from_numpy(y_test.to_numpy()).view(-1, 1)
X_train_torch.shape
y_train_torch.shape
#cell 6
class simpleNN():
  def __init__(self, X):
    self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
    self.bias = torch.rand(1, dtype=torch.float64, requires_grad=True)

  def forward(self, X):
    z = torch.matmul(X, self.weights) + self.bias
    y_pred = torch.sigmoid(z)
    return y_pred

  def loss_function(self, y_pred, y):
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()
    return loss
  #cell 7
  

# Training setup
learning_rate = 0.01   # smaller LR
epochs = 100

# Create model
model = simpleNN(X_train_torch)

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train_torch)

    # Compute loss
    loss = model.loss_function(y_pred, y_train_torch)

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    # Zero the gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#cell 8
with torch.no_grad():
  y_pred = model.forward(X_test_torch)
  y_pred = (y_pred > 0.5).float()
  accuracy = (y_pred == y_test_torch).float().mean()
  print(f"Accuracy: {accuracy.item()}")
  #cell 9
  import torch.nn as nn
import torch

class simpleNN(nn.Module):
  def __init__(self, X):
    super().__init__() # Call the parent class constructor
    self.weights = nn.Parameter(torch.rand(X.shape[1], 1, dtype=torch.float64))
    self.bias = nn.Parameter(torch.rand(1, dtype=torch.float64))

  def forward(self, X):
    z = torch.matmul(X, self.weights) + self.bias
    y_pred = torch.sigmoid(z)
    return y_pred

# Recreate the model instance after the class definition is updated
model = simpleNN(X_train_torch)

# Now you can save the model state dictionary
torch.save(model.state_dict(), 'simple_nn_model.pth')
print("Model saved successfully!")

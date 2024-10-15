import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing, datasets
from sklearn.metrics import mean_squared_error

# --- Importing and preparing the data ---#
housing = datasets.fetch_california_housing()

num_test = 10
scaler = preprocessing.StandardScaler()

X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = housing.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = housing.target[-num_test:]

# --- Implementing the neural network with Pytorch ------#
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train_torch = torch.from_numpy(X_train.astype(np.float32))
y_train_torch = torch.from_numpy(y_train.astype(np.float32))


def train_step(model, X_train, y_train, loss_function, optimizer):
    pred_train = model(X_train)
    loss = loss_function(pred_train, y_train)

    model.zero_grad()
    loss.backward()

    optimizer.step()
    return loss.item


for epoch in range(500):
    loss = train_step(model, X_train_torch, y_train_torch, loss_function, optimizer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - loss: {loss}")

# -----  Testing the model ----- #
X_test_torch = torch.from_numpy(X_test.astype(np.float32))
prediction = model(X_test_torch).detach().numpy()[:, 0]
print(
    "##################################################################################################"
)
print(f"Predictions: {prediction}")
print(f"Error: {mean_squared_error(y_test, prediction)}")
print(
    "##################################################################################################"
)

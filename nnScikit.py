from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing, datasets
from sklearn.metrics import mean_squared_error


nn_scikit = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    random_state=42,
    max_iter=2000,
)

housing = datasets.fetch_california_housing()

num_test = 10  # the last 10 samples as testing set.

scaler = preprocessing.StandardScaler()

X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = housing.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = housing.target[-num_test:]


nn_scikit.fit(X_train, y_train.ravel())
predictions = nn_scikit.predict(X_test)
print("Predictions", predictions, "\n")
print(mean_squared_error(y_test, predictions))

CONST = 0

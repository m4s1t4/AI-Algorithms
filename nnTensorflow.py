from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing, datasets
from sklearn.metrics import mean_squared_error

tf.random.set_seed(42)

housing = datasets.fetch_california_housing()

num_test = 10  # the last 10 samples as testing set.

scaler = preprocessing.StandardScaler()

X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = housing.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = housing.target[-num_test:]


# Keras Sequential model, is a list of layers instances to the constructor, including two fully connected hidden layers with 16 nodes and 8 nodes, respectively.

model = keras.Sequential(
    [
        keras.layers.Dense(units=16, activation="relu"),
        keras.layers.Dense(units=8, activation="relu"),
        keras.layers.Dense(units=1),
    ]
)

# We compile the model using Adam as the optimizer with a lerning rete of 0.01 and MSE as the learning goal.

model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.01))

model.fit(X_train, y_train, epochs=300)

predictions = model.predict(X_test)[:, 0]
print("\n")
print("Predictions: ", predictions, "\n")
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))

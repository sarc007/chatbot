import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential(
    [
        Dense(5, activation="relu"),
        Dense(10, activation="relu"),
        Dense(15),
    ]
)  # No weights to be added here

# Here we cannot check for weights
# model.weights

# Neither we can look at the summary
# model.summary()

# First we must call the model and evaluate it on test data
x = tf.ones((5, 20))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))